import logging
from azure.cosmos import CosmosClient, exceptions
from azure.cosmos.partition_key import PartitionKey
from datetime import datetime, timezone
 
class ReIDDatabase:
    def __init__(self, connection_string, database_name="occupancydb", container_name="person_features",
                 user_counts_container_name="user_counts", setup_container_name="setup-details",
                 user_logs_container_name="user_logs"):
        self.client = CosmosClient.from_connection_string(connection_string)
       
        # Ensure database creation
        self.database = self.client.create_database_if_not_exists(id=database_name)
        self.setup_container_name = setup_container_name
       
        # Create or access containers
        self.container = self.database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/person_id"),
            indexing_policy={
                'indexingMode': 'consistent',
                'automatic': True,
                'includedPaths': [{'path': '/*'}],
                'excludedPaths': [{'path': '/features/*'}]
            }
        )
       
        self.user_counts_container = self.database.create_container_if_not_exists(
            id=user_counts_container_name,
            partition_key=PartitionKey(path="/user_id"),
        )
       
        self.setup_container = self.database.create_container_if_not_exists(
            id=self.setup_container_name,
            partition_key=PartitionKey(path="/id"),
        )
       
        self.user_logs_container = self.database.create_container_if_not_exists(
            id=user_logs_container_name,
            partition_key=PartitionKey(path="/user_id"),
        )
       
        self.setup_logging()
 
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('reid_database.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
 
    async def get_all_user_documents(self):
        """Retrieve all user documents from the database."""
        try:
            query = "SELECT * FROM c"
            items = list(self.setup_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            return items
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve user documents: {str(e)}")
            return []
 
    def update_user_counts(self, user_id, camera_id, entry_count, exit_count):
        """Update user counts for a specific camera in a single document per user."""
        try:
            # Try to get existing document
            query = f"SELECT * FROM c WHERE c.id = '{user_id}'"
            items = list(self.user_counts_container.query_items(query, enable_cross_partition_query=True))
           
            timestamp = datetime.now(timezone.utc).isoformat()
           
            if items:
                # Update existing document
                doc = items[0]
                if "cameras" not in doc:
                    doc["cameras"] = {}
               
                doc["cameras"][camera_id] = {
                    "entry_count": entry_count,
                    "exit_count": exit_count,
                    "timestamp": timestamp
                }
                doc["last_updated"] = timestamp
               
                self.user_counts_container.replace_item(doc["id"], doc)
            else:
                # Create new document
                doc = {
                    "id": user_id,
                    "user_id": user_id,
                    "cameras": {
                        camera_id: {
                            "entry_count": entry_count,
                            "exit_count": exit_count,
                            "timestamp": timestamp
                        }
                    },
                    "last_updated": timestamp
                }
               
                self.user_counts_container.create_item(doc)
               
            self.logger.info(f"Updated counts for user {user_id}, camera {camera_id}")
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to update user counts: {str(e)}")
            return False
 
    def get_user_counts(self, user_id):
        """Retrieve counts for a specific user."""
        try:
            query = f"SELECT * FROM c WHERE c.id = '{user_id}'"
            items = list(self.user_counts_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
           
            if items:
                return items[0]
            return {"user_id": user_id, "cameras": {}}
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve user counts: {str(e)}")
            return {"user_id": user_id, "cameras": {}}
 
    def get_user_total_counts(self, user_id):
        """Get total entry/exit counts across all cameras for a user."""
        try:
            user_data = self.get_user_counts(user_id)
           
            total_entry = 0
            total_exit = 0
            camera_count = 0
           
            if "cameras" in user_data:
                for camera_id, camera_data in user_data["cameras"].items():
                    total_entry += camera_data.get("entry_count", 0)
                    total_exit += camera_data.get("exit_count", 0)
                    camera_count += 1
           
            return {
                "total_entry": total_entry,
                "total_exit": total_exit,
                "camera_count": camera_count
            }
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve total counts for user {user_id}: {str(e)}")
            return {"total_entry": 0, "total_exit": 0, "camera_count": 0}
 
    def store_user_log(self, user_id, camera_id, person_id, event_type, timestamp=None):
        """Store logs as a single document per user in the logs container."""
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc).isoformat()
 
            # Try to get existing document
            query = f"SELECT * FROM c WHERE c.id = '{user_id}'"
            items = list(self.user_logs_container.query_items(query, enable_cross_partition_query=True))
           
            log_entry = {
                "camera_id": camera_id,
                "person_id": person_id,
                "event_type": event_type,
                "timestamp": timestamp
            }
           
            if items:
                # Update existing document
                doc = items[0]
                if "logs" not in doc:
                    doc["logs"] = []
               
                doc["logs"].append(log_entry)
                doc["last_updated"] = timestamp
               
                self.user_logs_container.replace_item(doc["id"], doc)
            else:
                # Create new document
                doc = {
                    "id": user_id,
                    "user_id": user_id,
                    "logs": [log_entry],
                    "last_updated": timestamp
                }
               
                self.user_logs_container.create_item(doc)
               
            self.logger.info(f"Updated log for user {user_id}, camera {camera_id}")
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to store user log: {str(e)}")
            return False
 
    def get_user_logs(self, user_id, limit=100):
        """Retrieve logs for a specific user with optional limit."""
        try:
            query = f"SELECT * FROM c WHERE c.id = '{user_id}'"
            items = list(self.user_logs_container.query_items(query, enable_cross_partition_query=True))
           
            if items:
                doc = items[0]
                if "logs" in doc:
                    # Sort logs by timestamp (newest first) and apply limit
                    logs = sorted(doc["logs"], key=lambda x: x.get("timestamp", ""), reverse=True)
                    return logs[:limit] if limit else logs
                return []
            return []
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve logs for user {user_id}: {str(e)}")
            return []
 
    def get_camera_setup_details(self):
        """Retrieve camera setup details from the setup-details container."""
        try:
            query = "SELECT * FROM c"
            items = list(self.setup_container.query_items(query, enable_cross_partition_query=True))
            return items
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve camera setup details: {str(e)}")
            return []
 
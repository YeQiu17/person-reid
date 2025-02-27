import logging
from azure.cosmos import CosmosClient, exceptions
from azure.cosmos.partition_key import PartitionKey
from datetime import datetime, timedelta, timezone
import numpy as np

class ReIDDatabase:
    def __init__(self, connection_string, database_name="reid_db", container_name="person_features", 
                 counts_container_name="enter_exit_count", setup_container_name="setup-details", 
                 logs_container_name="logs", user_container_name="user_counts"):
        self.client = CosmosClient.from_connection_string(connection_string)
        
        # Ensure database creation
        self.database = self.client.create_database_if_not_exists(id=database_name)
        self.counts_container_name = counts_container_name
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
        
        self.counts_container = self.database.create_container_if_not_exists(
            id=self.counts_container_name,
            partition_key=PartitionKey(path="/camera_id"),
        )
        
        self.setup_container = self.database.create_container_if_not_exists(
            id=self.setup_container_name,
            partition_key=PartitionKey(path="/id"),
        )
        
        self.logs_container = self.database.create_container_if_not_exists(
            id=logs_container_name,
            partition_key=PartitionKey(path="/camera_id"),
        )
        
        self.user_container = self.database.create_container_if_not_exists(
            id=user_container_name,
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

    def update_user_counts(self, document):
        """Update or create user counts document."""
        try:
            # Create a unique ID combining user_id and camera_id
            document_id = f"{document['user_id']}_{document['camera_id']}"
            document['id'] = document_id

            # Upsert the document
            self.user_container.upsert_item(body=document)
            self.logger.info(f"Updated counts for user {document['user_id']}, camera {document['camera_id']}")
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to update user counts: {str(e)}")
            return False

    def get_user_counts(self, user_id):
        """Retrieve counts for a specific user."""
        try:
            
            query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
            items = list(self.user_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            return items
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve user counts: {str(e)}")
            return []

    def store_aggregated_log(self, camera_id, person_id, event_type, timestamp=None):
        """Store logs as a single aggregated document in the logs container."""
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc).isoformat()

            existing_doc = self.get_aggregated_logs(camera_id)

            if not existing_doc:
                existing_doc = {
                    "id": camera_id,
                    "camera_id": camera_id,
                    "logs": []
                }

            existing_doc["logs"].append({
                "person_id": person_id,
                "event_type": event_type,
                "timestamp": timestamp
            })

            self.logs_container.upsert_item(existing_doc)
            self.logger.info(f"Updated aggregated log for camera {camera_id}, person {person_id}, event {event_type}")
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to store aggregated log: {str(e)}")
            return False

    def get_aggregated_logs(self, camera_id):
        """Retrieve the aggregated log document for a specific camera."""
        try:
            query = f"SELECT * FROM c WHERE c.id = '{camera_id}'"
            items = list(self.logs_container.query_items(query, enable_cross_partition_query=True))

            if items:
                return items[0]
            return None
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve aggregated logs for camera {camera_id}: {str(e)}")
            return None

    def get_camera_setup_details(self):
        """Retrieve camera setup details from the setup-details container."""
        try:
            query = "SELECT * FROM c"
            items = self.setup_container.query_items(query, enable_cross_partition_query=True)
            for item in items:
                return item
            return {}
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve camera setup details: {str(e)}")
            return {}

    def get_all_counts(self):
        """Retrieve all entry/exit counts for all cameras."""
        try:
            counts = {}
            query = "SELECT * FROM c"
            items = self.counts_container.query_items(query, enable_cross_partition_query=True)

            for item in items:
                counts[str(item["camera_id"])] = {
                    "entry": int(item.get("entry", 0)),
                    "exit": int(item.get("exit", 0))
                }
            return counts
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve all counts: {str(e)}")
            return {}

    def update_counts(self, camera_id, entry_count, exit_count):
        """Update entry/exit counts for a specific camera."""
        try:
            item = {
                "id": str(camera_id),
                "camera_id": str(camera_id),
                "entry": int(entry_count),
                "exit": int(exit_count)
            }
            self.counts_container.upsert_item(body=item)
            self.logger.info(f"Updated counts for camera {camera_id}")
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to update counts: {str(e)}")
            return False
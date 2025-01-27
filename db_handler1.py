import logging
from azure.cosmos import CosmosClient, exceptions
from azure.cosmos.partition_key import PartitionKey
from datetime import datetime, timedelta, timezone
import numpy as np


class ReIDDatabase:
    def __init__(self, connection_string, database_name="occupancydb", container_name="person_features", 
                 counts_container_name="enter_exit_count", setup_container_name="setup-details", 
                 logs_container_name="logs"):
        self.client = CosmosClient.from_connection_string(connection_string)
        self.database = self.client.create_database_if_not_exists(id=database_name)
        self.counts_container_name = counts_container_name
        self.setup_container_name = setup_container_name
        
        # Create containers with organization_id as partition key
        self.container = self.database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/organization_id"),
            indexing_policy={
                'indexingMode': 'consistent',
                'automatic': True,
                'includedPaths': [{'path': '/*'}],
                'excludedPaths': [{'path': '/persons/*/features/*'}]
            }
        )
        self.counts_container = self.database.create_container_if_not_exists(
            id=self.counts_container_name,
            partition_key=PartitionKey(path="/organization_id"),
        )
        self.setup_container = self.database.create_container_if_not_exists(
            id=self.setup_container_name,
            partition_key=PartitionKey(path="/organization_id"),
        )
        self.logs_container = self.database.create_container_if_not_exists(
            id=logs_container_name,
            partition_key=PartitionKey(path="/organization_id"),
        )
        self.setup_logging()
        self.organization_id = None

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('reid_database.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def initialize_organization(self):
        """Get organization ID from setup details and initialize if needed."""
        try:
            setup_details = self.get_camera_setup_details()
            if setup_details:
                self.organization_id = setup_details.get('organization_id')
                return self.organization_id
            return None
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to initialize organization: {str(e)}")
            return None

    def _get_or_create_document(self, container, doc_type):
        """Helper method to get or create organization document."""
        if not self.organization_id:
            self.initialize_organization()
            if not self.organization_id:
                raise ValueError("Organization ID not initialized")

        try:
            query = f"SELECT * FROM c WHERE c.organization_id = '{self.organization_id}'"
            items = list(container.query_items(query, enable_cross_partition_query=True))
            
            if items:
                return items[0]
            
            # Create new document if none exists
            new_doc = {
                "id": self.organization_id,
                "organization_id": self.organization_id,
                "type": doc_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            
            if doc_type == "person_features":
                new_doc["persons"] = {}
            elif doc_type == "counts":
                new_doc["cameras"] = {}
            elif doc_type == "logs":
                new_doc["logs"] = []
                
            return container.create_item(body=new_doc)
            
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to get/create {doc_type} document: {str(e)}")
            return None

    def store_person(self, person_id, features, camera_id, timestamp=None):
        """Store person features in the organization's document."""
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc).isoformat()

            doc = self._get_or_create_document(self.container, "person_features")
            if not doc:
                return False

            if "persons" not in doc:
                doc["persons"] = {}

            doc["persons"][str(person_id)] = {
                "features": self._serialize_features(features),
                "camera_id": camera_id,
                "timestamp": timestamp,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            self.container.replace_item(item=doc["id"], body=doc)
            self.logger.info(f"Stored features for person {person_id} from camera {camera_id}")
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to store person data: {str(e)}")
            return False

    def get_person_features(self, person_id):
        """Retrieve features for a person from the organization's document."""
        try:
            doc = self._get_or_create_document(self.container, "person_features")
            if not doc or "persons" not in doc:
                return None

            person_data = doc["persons"].get(str(person_id))
            if person_data:
                return self._deserialize_features(person_data["features"])
            return None
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve person features: {str(e)}")
            return None

    def get_all_persons(self):
        """Retrieve all persons from the organization's document."""
        try:
            doc = self._get_or_create_document(self.container, "person_features")
            if not doc or "persons" not in doc:
                return {}

            persons = {}
            for person_id, person_data in doc["persons"].items():
                persons[person_id] = self._deserialize_features(person_data["features"])
            return persons
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve all persons: {str(e)}")
            return {}

    def store_aggregated_log(self, camera_id, person_id, event_type, timestamp=None):
        """Store logs in the organization's document."""
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc).isoformat()

            doc = self._get_or_create_document(self.logs_container, "logs")
            if not doc:
                return False

            if "logs" not in doc:
                doc["logs"] = []

            doc["logs"].append({
                "camera_id": camera_id,
                "person_id": person_id,
                "event_type": event_type,
                "timestamp": timestamp
            })

            self.logs_container.replace_item(item=doc["id"], body=doc)
            self.logger.info(f"Updated log for camera {camera_id}, person {person_id}, event {event_type}")
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to store log: {str(e)}")
            return False

    def get_aggregated_logs(self, camera_id=None):
        """Retrieve logs from the organization's document."""
        try:
            doc = self._get_or_create_document(self.logs_container, "logs")
            if not doc or "logs" not in doc:
                return []

            if camera_id:
                return [log for log in doc["logs"] if log["camera_id"] == camera_id]
            return doc["logs"]
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve logs: {str(e)}")
            return []

    def update_counts(self, camera_id, entry_count, exit_count):
        """Update counts in the organization's document."""
        try:
            doc = self._get_or_create_document(self.counts_container, "counts")
            if not doc:
                return False

            if "cameras" not in doc:
                doc["cameras"] = {}

            doc["cameras"][str(camera_id)] = {
                "entry": int(entry_count),
                "exit": int(exit_count),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            self.counts_container.replace_item(item=doc["id"], body=doc)
            self.logger.info(f"Updated counts for camera {camera_id}")
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to update counts: {str(e)}")
            return False

    def get_all_counts(self):
        """Retrieve all counts from the organization's document."""
        try:
            doc = self._get_or_create_document(self.counts_container, "counts")
            if not doc or "cameras" not in doc:
                return {}

            counts = {}
            for camera_id, camera_data in doc["cameras"].items():
                counts[camera_id] = {
                    "entry": camera_data["entry"],
                    "exit": camera_data["exit"]
                }
            return counts
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve counts: {str(e)}")
            return {}

    def _serialize_features(self, features):
        """Convert numpy array to list for JSON serialization."""
        return features.tolist() if isinstance(features, np.ndarray) else features

    def _deserialize_features(self, features):
        """Convert list back to numpy array."""
        return np.array(features)

    def cleanup_old_records(self, days_to_keep=30):
        """Clean up old records from the organization's documents."""
        try:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).isoformat()
            
            # Clean up person features
            doc = self._get_or_create_document(self.container, "person_features")
            if doc and "persons" in doc:
                doc["persons"] = {
                    pid: pdata for pid, pdata in doc["persons"].items()
                    if pdata["timestamp"] >= cutoff_date
                }
                self.container.replace_item(item=doc["id"], body=doc)

            # Clean up logs
            logs_doc = self._get_or_create_document(self.logs_container, "logs")
            if logs_doc and "logs" in logs_doc:
                logs_doc["logs"] = [
                    log for log in logs_doc["logs"]
                    if log["timestamp"] >= cutoff_date
                ]
                self.logs_container.replace_item(item=logs_doc["id"], body=logs_doc)

            self.logger.info(f"Cleaned up records older than {days_to_keep} days")
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to clean up old records: {str(e)}")
            return False
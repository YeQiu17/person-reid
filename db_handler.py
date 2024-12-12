# db_handler.py

import logging
from azure.cosmos import CosmosClient, exceptions
from azure.cosmos.partition_key import PartitionKey
from datetime import datetime, timedelta
import numpy as np


class ReIDDatabase:
    def __init__(self, connection_string, database_name="occupancydb", container_name="person_features", counts_container_name="enter_exit_count", setup_container_name="setup-details"):
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
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('reid_database.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def get_camera_setup_details(self):
        """Retrieve camera setup details from the `setup-details` container."""
        try:
            query = "SELECT * FROM c"
            items = self.setup_container.query_items(query, enable_cross_partition_query=True)
            for item in items:
                return item  # Assuming there is one setup detail document
            return {}
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve camera setup details: {str(e)}")
            return {}

    def _serialize_features(self, features):
        """Convert numpy array to list for JSON serialization."""
        return features.tolist() if isinstance(features, np.ndarray) else features

    def _deserialize_features(self, features):
        """Convert list back to numpy array."""
        return np.array(features)

    def store_person(self, person_id, features, camera_id, timestamp=None):
        """Store person features and metadata in Cosmos DB."""
        try:
            if timestamp is None:
                timestamp = datetime.utcnow().isoformat()

            document = {
                'id': f"{person_id}_{timestamp}",
                'person_id': str(person_id),
                'features': self._serialize_features(features),
                'camera_id': camera_id,
                'timestamp': timestamp,
                'last_updated': datetime.utcnow().isoformat()
            }
            self.container.create_item(document)
            self.logger.info(f"Stored features for person {person_id} from camera {camera_id}")
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to store person data: {str(e)}")
            return False

    def get_person_features(self, person_id):
        """Retrieve latest features for a person."""
        try:
            query = f"SELECT * FROM c WHERE c.person_id = '{person_id}' ORDER BY c.timestamp DESC OFFSET 0 LIMIT 1"
            items = self.container.query_items(query=query, enable_cross_partition_query=True)

            for item in items:
                return self._deserialize_features(item['features'])
            return None
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve person features: {str(e)}")
            return None

    def get_all_persons(self):
        """Retrieve all unique person IDs and their latest features."""
        try:
            query = """
            SELECT VALUE {
                'person_id': c.person_id,
                'features': c.features
            }
            FROM c
            WHERE NOT IS_NULL(c.features)
            """
            persons = {}
            items = self.container.query_items(query=query, enable_cross_partition_query=True)

            for item in items:
                persons[item['person_id']] = self._deserialize_features(item['features'])
            return persons
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve all persons: {str(e)}")
            return {}

    def update_person_features(self, person_id, new_features, camera_id, threshold=0.9):
        """Update person features only if significant variation is detected."""
        try:
            current_features = self.get_person_features(person_id)
            if current_features is not None:
                similarity = np.dot(current_features, new_features)
                if similarity > threshold:
                    self.logger.info(f"Skipping update for person {person_id}, similarity={similarity}")
                    return False

            timestamp = datetime.utcnow().isoformat()
            self.store_person(person_id, new_features, camera_id, timestamp)
            self.logger.info(f"Updated features for person {person_id} from camera {camera_id}")
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to update person features: {str(e)}")
            return False

    def cleanup_old_records(self, days_to_keep=30):
        """Remove records older than the specified number of days."""
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
            query = f"SELECT c.id, c.person_id FROM c WHERE c.timestamp < '{cutoff_date}'"

            items = self.container.query_items(query=query, enable_cross_partition_query=True)
            for item in items:
                self.container.delete_item(item['id'], partition_key=item['person_id'])
            self.logger.info(f"Cleaned up records older than {days_to_keep} days")
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to clean up old records: {str(e)}")
            return False

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

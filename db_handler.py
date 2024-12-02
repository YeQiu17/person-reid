# db_handler.py

import asyncio
from azure.cosmos import CosmosClient, exceptions
import logging
from azure.cosmos.partition_key import PartitionKey
from datetime import datetime
import numpy as np
import json
from datetime import datetime, timedelta


class ReIDDatabase:
    def __init__(self, connection_string, database_name="reid_db", container_name="person_features",counts_container_name='enter_exit_count'):
        self.client = CosmosClient.from_connection_string(connection_string)
        
        # Ensure database creation
        self.database = self.client.create_database_if_not_exists(id=database_name)
        self.counts_container_name = counts_container_name
        # Create or access container without setting throughput (for serverless accounts)
        self.container = self.database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/person_id"),  # Properly define partition key
            indexing_policy={
                'indexingMode': 'consistent',
                'automatic': True,
                'includedPaths': [{'path': '/*'}],
                'excludedPaths': [{'path': '/features/*'}]  # Exclude indexing of large feature arrays
            }
        )
        self.counts_container = self.database.create_container_if_not_exists(
            id=self.counts_container_name,
            partition_key=PartitionKey(path="/camera_id"),
        )    
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('reid_database.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _serialize_features(self, features):
        """Convert numpy array to list for JSON serialization"""
        return features.tolist() if isinstance(features, np.ndarray) else features

    def _deserialize_features(self, features):
        """Convert list back to numpy array"""
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

            # Removed 'await' since `create_item` is synchronous
            self.container.create_item(document)
            self.logger.info(f"Stored features for person {person_id} from camera {camera_id}")
            return True

        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to store person data: {str(e)}")
            return False


    async def get_person_features(self, person_id):
        """Retrieve latest features for a person"""
        try:
            query = f"SELECT * FROM c WHERE c.person_id = '{person_id}' ORDER BY c.timestamp DESC OFFSET 0 LIMIT 1"
            items = self.container.query_items(query=query, enable_cross_partition_query=True)
            
            async for item in items:
                return self._deserialize_features(item['features'])
            
            return None

        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve person features: {str(e)}")
            return None

    async def get_all_persons(self):
        """Retrieve all unique person IDs and their latest features."""
        try:
            query = """
            SELECT DISTINCT VALUE {
                'person_id': c.person_id,
                'features': c.features
            }
            FROM c
            WHERE c.timestamp IN (
                SELECT VALUE MAX(d.timestamp)
                FROM d
                WHERE d.person_id = c.person_id
                GROUP BY d.person_id
            )
            """
            
            persons = {}
            items = self.container.query_items(query=query, enable_cross_partition_query=True)
            
            # Synchronous iteration over ItemPaged
            for item in items:
                persons[item['person_id']] = self._deserialize_features(item['features'])
            
            return persons

        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to retrieve all persons: {str(e)}")
            return {}


    async def update_person_features(self, person_id, new_features, camera_id):
        """Update person features with new observation"""
        try:
            timestamp = datetime.utcnow().isoformat()
            
            # Store new observation
            self.store_person(person_id, new_features, camera_id, timestamp)
            
            self.logger.info(f"Updated features for person {person_id} from camera {camera_id}")
            return True

        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to update person features: {str(e)}")
            return False

    async def cleanup_old_records(self, days_to_keep=30):
        """Remove records older than specified days"""
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
            query = f"SELECT * FROM c WHERE c.timestamp < '{cutoff_date}'"
            
            items = self.container.query_items(query=query, enable_cross_partition_query=True)
            async for item in items:
                self.container.delete_item(item['id'], partition_key=item['person_id'])
            
            self.logger.info(f"Cleaned up records older than {days_to_keep} days")
            return True

        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"Failed to cleanup old records: {str(e)}")
            return False
    
    async def get_all_counts(self):
        """Retrieve all entry/exit counts for all cameras from the database."""
        counts = {}
        query = "SELECT * FROM c"
        items = self.counts_container.query_items(query, enable_cross_partition_query=True)
        for item in items:
            camera_id = item["camera_id"]
            counts[camera_id] = {
                "entry": item.get("entry", 0),
                "exit": item.get("exit", 0)
            }
        return counts

    async def update_counts(self, camera_id, entry_count, exit_count):
        """Update entry/exit counts for a specific camera."""
        camera_id_str = str(camera_id)  # Ensure ID is a string
        item = {
            "id": camera_id_str,  # Ensure ID is a string
            "camera_id": camera_id_str,
            "entry": entry_count,
            "exit": exit_count
        }
        self.counts_container.upsert_item(body=item)

from msal import ConfidentialClientApplication
import jwt
from azure.cosmos import CosmosClient, exceptions
from azure.cosmos.partition_key import PartitionKey
from datetime import datetime,timezone
import logging
import numpy as np

class B2CAuthenticator:
    def __init__(self, tenant_id, client_id, client_secret, policy_name):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.policy_name = policy_name
        self.authority = f"https://{tenant_id}.b2clogin.com/{tenant_id}.onmicrosoft.com/{policy_name}"
        
        self.app = ConfidentialClientApplication(
            client_id=client_id,
            client_credential=client_secret,
            authority=self.authority
        )
        
    def validate_token(self, token):
        try:
            decoded_token = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            return {
                'user_id': decoded_token.get('oid'),
                'org_id': decoded_token.get('extension_OrganizationId')
            }
        except jwt.InvalidTokenError:
            return None

class OrganizationPersonFeatures:
    def __init__(self, connection_string, b2c_authenticator, database_name="occupancydb"):
        self.client = CosmosClient.from_connection_string(connection_string)
        self.b2c_authenticator = b2c_authenticator
        self.database = self.client.create_database_if_not_exists(id=database_name)
        
        # Container for consolidated person features per organization
        self.person_features_container = self.database.create_container_if_not_exists(
            id="org_person_features",
            partition_key=PartitionKey(path="/org_id")
        )
        
        # Container for entry/exit counts
        self.counts_container = self.database.create_container_if_not_exists(
            id="org_counts",
            partition_key=PartitionKey(path="/org_id")
        )
        
        # Container for activity logs
        self.logs_container = self.database.create_container_if_not_exists(
            id="org_logs",
            partition_key=PartitionKey(path="/org_id")
        )
        
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('org_person_features.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_auth_info(self, token):
        auth_info = self.b2c_authenticator.validate_token(token)
        if not auth_info:
            raise ValueError("Invalid token or missing organization ID")   
        return auth_info

    def _serialize_features(self, features):
        return features.tolist() if isinstance(features, np.ndarray) else features

    def _deserialize_features(self, features):
        return np.array(features)

    def get_org_document(self, token):
        """Get or create organization's consolidated document."""
        try:
            auth_info = self.get_auth_info(token)
            org_id = auth_info['org_id']
            
            try:
                org_doc = self.person_features_container.read_item(
                    item=org_id,
                    partition_key=org_id
                )
            except exceptions.CosmosResourceNotFoundError:
                # Initialize new organization document
                org_doc = {
                    'id': org_id,
                    'org_id': org_id,
                    'persons': {},  # Dictionary of person_id -> person data
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }
                self.person_features_container.create_item(body=org_doc)
            
            return org_doc
        except Exception as e:
            self.logger.error(f"Failed to get organization document: {str(e)}")
            raise

    def store_person_features(self, token, person_id, features, camera_id):
        """Store person features in the organization's consolidated document."""
        try:
            org_doc = self.get_org_document(token)
            
            # Create or update person entry
            person_data = org_doc['persons'].get(person_id, {
                'features_history': [],
                'created_at': datetime.now(timezone.utc).isoformat()
            })
            
            # Add new features
            person_data['features_history'].append({
                'features': self._serialize_features(features),
                'camera_id': camera_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            # Keep only the last 10 feature entries
            if len(person_data['features_history']) > 10:
                person_data['features_history'] = person_data['features_history'][-10:]
            
            org_doc['persons'][person_id] = person_data
            org_doc['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            self.person_features_container.replace_item(
                item=org_doc['id'],
                body=org_doc
            )
            
            self.logger.info(f"Stored features for person {person_id} in organization {org_doc['org_id']}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store person features: {str(e)}")
            return False

    def get_person_features(self, token, person_id):
        """Get latest features for a person from organization document."""
        try:
            org_doc = self.get_org_document(token)
            person_data = org_doc['persons'].get(person_id)
            
            if not person_data or not person_data['features_history']:
                return None
                
            # Return the most recent features
            latest_features = person_data['features_history'][-1]['features']
            return self._deserialize_features(latest_features)
        except Exception as e:
            self.logger.error(f"Failed to get person features: {str(e)}")
            return None

    def update_counts(self, token, camera_id, counts_data):
        """Update entry/exit counts for the organization."""
        try:
            auth_info = self.get_auth_info(token)
            org_id = auth_info['org_id']
            
            counts_doc = {
                'id': f"{org_id}_counts",
                'org_id': org_id,
                'camera_counts': {
                    camera_id: {
                        'entry': counts_data['entry'],
                        'exit': counts_data['exit'],
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                },
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            self.counts_container.upsert_item(body=counts_doc)
            self.logger.info(f"Updated counts for camera {camera_id} in organization {org_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update counts: {str(e)}")
            return False

    def store_log(self, token, person_id, event_type, camera_id):
        """Store activity log for the organization."""
        try:
            auth_info = self.get_auth_info(token)
            org_id = auth_info['org_id']
            
            log_doc = {
                'id': f"{org_id}_logs",
                'org_id': org_id,
                'logs': [{
                    'person_id': person_id,
                    'event_type': event_type,
                    'camera_id': camera_id,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }],
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            self.logs_container.upsert_item(body=log_doc)
            self.logger.info(f"Stored log for person {person_id} in organization {org_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store log: {str(e)}")
            return False

    def get_all_person_features(self, token):
        """Get all person features for the organization."""
        try:
            org_doc = self.get_org_document(token)
            
            all_features = {}
            for person_id, person_data in org_doc['persons'].items():
                if person_data['features_history']:
                    latest_features = person_data['features_history'][-1]['features']
                    all_features[person_id] = self._deserialize_features(latest_features)
            
            return all_features
        except Exception as e:
            self.logger.error(f"Failed to get all person features: {str(e)}")
            return {}
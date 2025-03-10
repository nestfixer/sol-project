"""
Knowledge Base module for the Solana Token Analysis Agent Swarm.
Serves as a central repository for sharing data between agents.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Tuple

from loguru import logger


class EntryType(Enum):
    """Enum representing the types of entries in the knowledge base."""
    TOKEN_DATA = "token_data"
    TRANSACTION_DATA = "transaction_data"
    PATTERN = "pattern"
    RISK_ASSESSMENT = "risk_assessment"
    INVESTMENT_OPPORTUNITY = "investment_opportunity"
    SYSTEM_CONFIG = "system_config"
    AGENT_STATUS = "agent_status"
    CACHED_API_RESPONSE = "cached_api_response"
    

class KnowledgeBaseEntry:
    """Represents a single entry in the knowledge base."""
    
    def __init__(
        self,
        entry_id: str,
        entry_type: EntryType,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,  # Time to live in seconds
        source_agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.id = entry_id
        self.type = entry_type
        self.data = data
        self.metadata = metadata or {}
        self.ttl = ttl
        self.source_agent_id = source_agent_id
        self.tags = set(tags or [])
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        self.expires_at = None
        
        if ttl is not None:
            # Convert TTL to expiration timestamp
            expiry_time = time.time() + ttl
            self.expires_at = datetime.fromtimestamp(expiry_time).isoformat()
    
    def update(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update the entry with new data and metadata."""
        self.data = data
        if metadata:
            self.metadata.update(metadata)
        self.updated_at = datetime.utcnow().isoformat()
    
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.expires_at is None:
            return False
        
        expiry_time = datetime.fromisoformat(self.expires_at)
        return datetime.utcnow() > expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entry to a dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "data": self.data,
            "metadata": self.metadata,
            "ttl": self.ttl,
            "source_agent_id": self.source_agent_id,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeBaseEntry":
        """Create a KnowledgeBaseEntry from a dictionary."""
        entry = cls(
            entry_id=data["id"],
            entry_type=EntryType(data["type"]),
            data=data["data"],
            metadata=data.get("metadata", {}),
            ttl=data.get("ttl"),
            source_agent_id=data.get("source_agent_id"),
            tags=data.get("tags", []),
        )
        entry.created_at = data["created_at"]
        entry.updated_at = data["updated_at"]
        entry.expires_at = data.get("expires_at")
        return entry


class KnowledgeBase:
    """
    Central knowledge repository for the agent swarm.
    Provides storage, retrieval, and querying capabilities.
    """
    
    def __init__(self, persistence_path: Optional[str] = None):
        """
        Initialize the knowledge base.
        
        Args:
            persistence_path: Optional path to persist the knowledge base.
                If provided, the knowledge base will be loaded from and saved to this path.
        """
        self.entries: Dict[str, KnowledgeBaseEntry] = {}
        self.persistence_path = persistence_path
        self.indices: Dict[str, Dict[str, Set[str]]] = {
            "type": {},       # type -> set of entry_ids
            "source": {},     # source_agent_id -> set of entry_ids
            "tag": {},        # tag -> set of entry_ids
        }
        self.lock = asyncio.Lock()
        self.last_cleanup = time.time()
        self.cleanup_interval = 60  # Clean up expired entries every 60 seconds
        
        # Load persisted data if available
        if persistence_path and os.path.exists(persistence_path):
            self._load_from_file()
    
    async def add_entry(
        self,
        entry_id: str,
        entry_type: EntryType,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        source_agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Add a new entry to the knowledge base.
        
        Args:
            entry_id: Unique identifier for the entry
            entry_type: Type of the entry
            data: The data to store
            metadata: Optional metadata about the entry
            ttl: Optional time-to-live in seconds
            source_agent_id: Optional ID of the agent that created the entry
            tags: Optional list of tags to associate with the entry
            
        Returns:
            The ID of the created entry
        """
        async with self.lock:
            # Check if entry already exists
            if entry_id in self.entries:
                # Update existing entry
                self.entries[entry_id].update(data, metadata)
                
                # Update TTL if provided
                if ttl is not None:
                    self.entries[entry_id].ttl = ttl
                    expiry_time = time.time() + ttl
                    self.entries[entry_id].expires_at = datetime.fromtimestamp(expiry_time).isoformat()
                
                # Update tags if provided
                if tags:
                    # Remove old tag indices
                    for tag in self.entries[entry_id].tags:
                        if tag in self.indices["tag"]:
                            self.indices["tag"][tag].discard(entry_id)
                    
                    # Add new tags
                    self.entries[entry_id].tags = set(tags)
                    for tag in tags:
                        if tag not in self.indices["tag"]:
                            self.indices["tag"][tag] = set()
                        self.indices["tag"][tag].add(entry_id)
                
                logger.debug(f"Updated knowledge base entry: {entry_id}")
            else:
                # Create new entry
                entry = KnowledgeBaseEntry(
                    entry_id=entry_id,
                    entry_type=entry_type,
                    data=data,
                    metadata=metadata,
                    ttl=ttl,
                    source_agent_id=source_agent_id,
                    tags=tags,
                )
                self.entries[entry_id] = entry
                
                # Update indices
                # Type index
                if entry_type.value not in self.indices["type"]:
                    self.indices["type"][entry_type.value] = set()
                self.indices["type"][entry_type.value].add(entry_id)
                
                # Source index
                if source_agent_id:
                    if source_agent_id not in self.indices["source"]:
                        self.indices["source"][source_agent_id] = set()
                    self.indices["source"][source_agent_id].add(entry_id)
                
                # Tag indices
                if tags:
                    for tag in tags:
                        if tag not in self.indices["tag"]:
                            self.indices["tag"][tag] = set()
                        self.indices["tag"][tag].add(entry_id)
                
                logger.debug(f"Added new knowledge base entry: {entry_id}")
            
            # Persist if a path is set
            if self.persistence_path:
                await self._persist_to_file()
            
            # Clean up expired entries
            await self._maybe_cleanup_expired()
            
            return entry_id
    
    async def get_entry(self, entry_id: str) -> Optional[KnowledgeBaseEntry]:
        """
        Get an entry from the knowledge base by ID.
        
        Args:
            entry_id: The ID of the entry to retrieve
            
        Returns:
            The entry if found and not expired, None otherwise
        """
        async with self.lock:
            # Clean up expired entries
            await self._maybe_cleanup_expired()
            
            # Return the entry if it exists and hasn't expired
            if entry_id in self.entries and not self.entries[entry_id].is_expired():
                return self.entries[entry_id]
            return None
    
    async def query_entries(
        self,
        entry_type: Optional[EntryType] = None,
        source_agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        query_filter: Optional[callable] = None,
    ) -> List[KnowledgeBaseEntry]:
        """
        Query entries based on various criteria.
        
        Args:
            entry_type: Optional type of entries to query
            source_agent_id: Optional source agent ID to filter by
            tags: Optional list of tags to filter by (entries must have ALL tags)
            query_filter: Optional filter function that takes an entry and returns a boolean
            
        Returns:
            List of matching entries
        """
        async with self.lock:
            # Clean up expired entries
            await self._maybe_cleanup_expired()
            
            # Start with all entry IDs
            result_ids: Set[str] = set(self.entries.keys())
            
            # Filter by type
            if entry_type:
                type_ids = self.indices["type"].get(entry_type.value, set())
                result_ids &= type_ids
            
            # Filter by source agent
            if source_agent_id:
                source_ids = self.indices["source"].get(source_agent_id, set())
                result_ids &= source_ids
            
            # Filter by tags (entry must have ALL specified tags)
            if tags:
                for tag in tags:
                    tag_ids = self.indices["tag"].get(tag, set())
                    result_ids &= tag_ids
            
            # Get the entries
            results = [self.entries[entry_id] for entry_id in result_ids 
                      if not self.entries[entry_id].is_expired()]
            
            # Apply custom filter if provided
            if query_filter:
                results = [entry for entry in results if query_filter(entry)]
            
            return results
    
    async def delete_entry(self, entry_id: str) -> bool:
        """
        Delete an entry from the knowledge base.
        
        Args:
            entry_id: The ID of the entry to delete
            
        Returns:
            True if the entry was deleted, False if it wasn't found
        """
        async with self.lock:
            if entry_id in self.entries:
                entry = self.entries[entry_id]
                
                # Remove from indices
                if entry.type.value in self.indices["type"]:
                    self.indices["type"][entry.type.value].discard(entry_id)
                
                if entry.source_agent_id and entry.source_agent_id in self.indices["source"]:
                    self.indices["source"][entry.source_agent_id].discard(entry_id)
                
                for tag in entry.tags:
                    if tag in self.indices["tag"]:
                        self.indices["tag"][tag].discard(entry_id)
                
                # Remove the entry
                del self.entries[entry_id]
                
                # Persist if a path is set
                if self.persistence_path:
                    await self._persist_to_file()
                
                logger.debug(f"Deleted knowledge base entry: {entry_id}")
                return True
            
            return False
    
    async def clear(self) -> None:
        """Clear all entries from the knowledge base."""
        async with self.lock:
            self.entries.clear()
            for index_type in self.indices:
                self.indices[index_type].clear()
            
            # Persist empty knowledge base if a path is set
            if self.persistence_path:
                await self._persist_to_file()
            
            logger.info("Cleared knowledge base")
    
    async def _maybe_cleanup_expired(self) -> None:
        """Clean up expired entries if cleanup interval has elapsed."""
        current_time = time.time()
        if current_time - self.last_cleanup >= self.cleanup_interval:
            await self._cleanup_expired()
            self.last_cleanup = current_time
    
    async def _cleanup_expired(self) -> None:
        """Remove expired entries from the knowledge base."""
        # Find expired entries
        expired_ids = [
            entry_id for entry_id, entry in self.entries.items()
            if entry.is_expired()
        ]
        
        # Delete them
        for entry_id in expired_ids:
            await self.delete_entry(entry_id)
        
        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired entries")
    
    async def _persist_to_file(self) -> None:
        """Save the knowledge base to a file."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            
            # Convert entries to dictionaries
            entries_dict = {
                entry_id: entry.to_dict()
                for entry_id, entry in self.entries.items()
            }
            
            # Write to file
            with open(self.persistence_path, 'w') as f:
                json.dump(entries_dict, f)
            
            logger.debug(f"Persisted knowledge base to {self.persistence_path}")
        except Exception as e:
            logger.error(f"Failed to persist knowledge base: {str(e)}")
    
    def _load_from_file(self) -> None:
        """Load the knowledge base from a file."""
        try:
            with open(self.persistence_path, 'r') as f:
                entries_dict = json.load(f)
            
            # Clear current entries and indices
            self.entries.clear()
            for index_type in self.indices:
                self.indices[index_type].clear()
            
            # Load entries
            for entry_id, entry_data in entries_dict.items():
                entry = KnowledgeBaseEntry.from_dict(entry_data)
                
                # Skip expired entries
                if entry.is_expired():
                    continue
                
                self.entries[entry_id] = entry
                
                # Update indices
                # Type index
                if entry.type.value not in self.indices["type"]:
                    self.indices["type"][entry.type.value] = set()
                self.indices["type"][entry.type.value].add(entry_id)
                
                # Source index
                if entry.source_agent_id:
                    if entry.source_agent_id not in self.indices["source"]:
                        self.indices["source"][entry.source_agent_id] = set()
                    self.indices["source"][entry.source_agent_id].add(entry_id)
                
                # Tag indices
                for tag in entry.tags:
                    if tag not in self.indices["tag"]:
                        self.indices["tag"][tag] = set()
                    self.indices["tag"][tag].add(entry_id)
            
            logger.info(f"Loaded {len(self.entries)} entries from {self.persistence_path}")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")

from typing import Dict, Optional
from lmcache.compactor.core.block_manager import GranularBlockManager
from lmcache.compactor.h2o_local_compactor import H2OCompactor
from lmcache.compactor.sink_local_compactor import SinkCompactor
from lmcache.compactor.base_local_compactor import BaseLocalCompactor
from lmcache.compactor.utils import CompactorInput, CompactorOutput, CompactorMetadata
from lmcache.compactor.base_scheduler_compactor import BaseSchedulerCompactor

__all__ = ["H2OCompactor", "SinkCompactor"
           "BaseSchedulerCompactor",
           "CompactorInput", "CompactorOutput","CompactorMetadata",
           "GranularBlockManager"]

class LMCacheCompactorBuilder:
    _instances: Dict[str, BaseLocalCompactor] = {}

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        compactor_type = "H2O",
        compactor_metadata = None,
    ) -> BaseLocalCompactor:
        """
        Builds a new LMCacheCompactor instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        if instance_id not in cls._instances:
            if compactor_type == "H2O":
                compactor = H2OCompactor(compactor_metadata)
            elif compactor_type == "Sink":
                compactor = SinkCompactor(compactor_metadata)
            else:
                raise Exception(f"Compactor type {compactor_type} not supported")
            cls._instances[instance_id] = compactor
            return compactor
        else:
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[BaseLocalCompactor]:
        """Returns the LMCacheEngine instance associated with the instance ID, 
        or None if not found."""
        return cls._instances.get(instance_id)
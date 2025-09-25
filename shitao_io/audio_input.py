
from typing import List, Dict, Any
from typing import Callable, Optional

class AudioInput:
    def __init__(self) -> None:
        pass

    def get_available_devices(self) -> List[Dict[str, Any]]:
        '''
        Return a list of available audio input devices.
        
        
        '''
        # TODO
        pass

    def register_callback(self,device_id: int, callback:Callable,save_to_buffer:bool=False) -> None:
        # TODO
        pass
    
    def set_buffer_size(self,device_id: int, size:int) -> None:
        # TODO
        pass
    
    def start_stream(self,device_id: int) -> None:
        # TODO
        pass
    
    def stop_stream(self,device_id: int) -> None:
        # TODO
        pass
    
    def get_active_streams(self) -> List[int]:
        # TODO
        pass
    
    def test_device_delay(self,device_id: int) -> Optional[float]:
        # TODO
        pass
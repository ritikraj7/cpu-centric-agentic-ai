import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LocalEnvironmentConfig:
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 30


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)
        self.execution_log = []  # Store all command executions for logging

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None):
        """Execute a command in the local environment and return the result as a dict."""
        import time
        
        start_time = time.time()
        cwd = cwd or self.config.cwd or os.getcwd()
        timeout_val = timeout or self.config.timeout
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                cwd=cwd,
                env=os.environ | self.config.env,
                timeout=timeout_val,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            end_time = time.time()
            duration = end_time - start_time
            
            # Log the execution
            execution_record = {
                "command": command,
                "cwd": cwd,
                "timeout": timeout_val,
                "timestamp_start": start_time,
                "timestamp_end": end_time,
                "duration_seconds": duration,
                "returncode": result.returncode,
                "output": result.stdout,
                "status": "completed"
            }
            self.execution_log.append(execution_record)
            
            return {"output": result.stdout, "returncode": result.returncode}
            
        except subprocess.TimeoutExpired as e:
            end_time = time.time()
            duration = end_time - start_time
            
            # Log the timeout
            execution_record = {
                "command": command,
                "cwd": cwd,
                "timeout": timeout_val,
                "timestamp_start": start_time,
                "timestamp_end": end_time,
                "duration_seconds": duration,
                "returncode": -1,
                "output": str(e),
                "status": "timeout"
            }
            self.execution_log.append(execution_record)
            
            return {"output": f"Command timed out after {timeout_val}s: {command}", "returncode": -1}
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            # Log the error
            execution_record = {
                "command": command,
                "cwd": cwd,
                "timeout": timeout_val,
                "timestamp_start": start_time,
                "timestamp_end": end_time,
                "duration_seconds": duration,
                "returncode": -1,
                "output": str(e),
                "status": "error"
            }
            self.execution_log.append(execution_record)
            
            return {"output": f"Command failed: {str(e)}", "returncode": -1}

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | platform.uname()._asdict() | os.environ

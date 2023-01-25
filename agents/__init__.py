from .base import *
from .icarl import *
from .nil import *
from .nil_cpp import *
from .nil_global import *

def nil(model, use_amp, config, optimizer_regime, cuda, buffer_cuda, log_buffer, log_interval, batch_metrics):
    implementation = config.get("implementation", "")
    agent = nil_agent
    if implementation == "cpp":
        agent = nil_cpp_agent
    elif implementation == "local":
        agent = nil_local_agent
    return agent(model, use_amp, config, optimizer_regime, cuda, buffer_cuda, log_buffer, log_interval, batch_metrics)

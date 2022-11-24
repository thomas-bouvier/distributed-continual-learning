from .base import *
from .icarl import *
from .nil import *
from .nil_cpp import *
from .nil_global import *
from .nil_cpp_global import *

def nil(model, use_amp, config, optimizer_regime, cuda, buffer_cuda, log_buffer, log_interval, batch_metrics):
    implementation = config.get("implementation", "")
    agent = nil_agent
    if implementation == "cpp":
        agent = nil_cpp_agent
    elif implementation == "global":
        agent = nil_global_agent
    elif implementation == "cpp_global":
        agent = nil_cpp_global_agent
    return agent(model, use_amp, config, optimizer_regime, cuda, buffer_cuda, log_buffer, log_interval, batch_metrics)

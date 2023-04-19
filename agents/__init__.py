from .base import *

def nil(model, use_mask, use_amp, agent_config, optimizer_regime, batch_size, cuda, log_level, log_buffer, log_interval, batch_metrics):
    implementation = agent_config.get("implementation", "")
    if implementation == "cpp":
        from .nil_cpp import nil_cpp_agent
        agent = nil_cpp_agent
    elif implementation == "cpp_cat":
        from .nil_cpp_cat import nil_cpp_cat_agent
        agent = nil_cpp_cat_agent
    elif implementation == "local":
        from .nil_local import nil_local_agent
        agent = nil_local_agent
    else:
        from .nil import nil_agent
        agent = nil_agent
    return agent(model, use_mask, use_amp, agent_config, optimizer_regime, batch_size, cuda, log_level, log_buffer, log_interval, batch_metrics)

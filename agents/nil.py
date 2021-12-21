

def nil(model, config, optimizer, criterion, cuda, log_interval):
    implementation = config.get('implementation', '')
    if implementation == 'v4':
        return icarl_v4_agent(model, config, optimizer, criterion, cuda, log_interval)
    return nil_agent(model, config, optimizer, criterion, cuda, log_interval)

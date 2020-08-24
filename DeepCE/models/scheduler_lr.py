

def step_lr(step_pattern, step):
    if step >= step_pattern[3]:
        return 0.02
    elif step >= step_pattern[2]:
        return 0.002
    elif step >= step_pattern[1]:
        return 0.0002
    else:
        return 0.00002
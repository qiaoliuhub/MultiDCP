

def step_lr(step_pattern, step):
    if step >= step_pattern[3]:
        return 100
    elif step >= step_pattern[2]:
        return 10
    elif step >= step_pattern[1]:
        return 1
    else:
        return 0.1
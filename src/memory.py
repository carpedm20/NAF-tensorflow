class Memory(object):
  def __init__(self):
    self.prestates = []
    self.actions = []
    self.rewards = []
    self.terminals = []
    self.poststates = []

  def add(self, prestate, action, reward, terminal, poststate):
    self.prestates.append(prestate)
    self.actions.append(action)
    self.rewards.append(reward)
    self.terminals.append(terminal)
    self.poststates.append(poststate)

  @property
  def size(self):
    return len(self.rewards)


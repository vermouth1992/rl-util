"""
Due to the high computation requirement, we employ the following process:
- A process that collects data and performs learning. This controls the ratio between data collection and policy updates
- A process that samples data and performs dynamics training
- A process that samples data to perform rollouts
"""
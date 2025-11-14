# trainer_queue.py
import asyncio

# Shared queue used to hand off raw JSON cycles for training
training_queue = asyncio.Queue()

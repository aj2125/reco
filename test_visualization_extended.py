import pandas as pd
import random
import json
from datetime import datetime, timedelta

# Load reference tables
marginals_df = pd.read_csv('marginals.csv')
transitions_df = pd.read_csv('transitions.csv')
cooccurrence_df = pd.read_csv('cooccurrence.csv')
weights_df = pd.read_csv('event_weights.csv')
ymmt_df = pd.read_csv('cars_ymmt.csv')  # Contains vehicle metadata

# Create composite string identifier
ymmt_df['vehicle'] = ymmt_df.apply(lambda r: f"{int(r.year)} {r.make} {r.model} {r.trim}", axis=1)

# Load event fields mapping
with open('event_fields.json') as f:
    event_fields = json.load(f)

# Build dicts
marginals = marginals_df.set_index('event_type')['marginal_prob'].to_dict()
transitions = {}
for _, row in transitions_df.iterrows():
    transitions.setdefault(row['from_event'], {})[row['to_event']] = row['probability']
cooccurs = {}
for _, row in cooccurrence_df.iterrows():
    cooccurs.setdefault(row['event'], []).append((row['co_event'], row['probability']))
weights = weights_df.set_index('event_type')['weight'].to_dict()

# Configuration
NUM_USERS = 100
SESSIONS_PER_USER = 2
OUTPUT_PATH = 'mock_events.csv'

records = []
for u in range(1, NUM_USERS + 1):
    user = f'U{u:03d}'
    for _ in range(SESSIONS_PER_USER):
        # Sample first event by marginals
        first_event = random.choices(list(marginals.keys()), list(marginals.values()))[0]
        seq = [first_event]
        prev = first_event
        # Build sequence
        while prev in transitions and transitions[prev]:
            choices, probs = zip(*transitions[prev].items())
            next_event = random.choices(choices, probs)[0]
            seq.append(next_event)
            prev = next_event
        # Emit events with co-occurrence
        timestamp = datetime.now() - timedelta(hours=random.uniform(0, 168))
        i = 0
        while i < len(seq):
            ev = seq[i]
            # Co-occurrence injection
            if ev in cooccurs:
                for co_ev, p in cooccurs[ev]:
                    if random.random() < p:
                        seq.insert(i + 1, co_ev)
            # Create record
            rec = {'user_id': user, 'event_type': ev, 'timestamp': timestamp}
            fields = event_fields.get(ev, [])
            # Attach individual YMMT metadata if required
            if any(f in fields for f in ['year', 'make', 'model', 'trim']):
                vehicle = ymmt_df.sample(1).iloc[0]
                # individual fields
                rec['year'] = int(vehicle['year'])
                rec['make'] = vehicle['make']
                rec['model'] = vehicle['model']
                rec['trim'] = vehicle['trim']
                # composite
                rec['vehicle'] = vehicle['vehicle']
                if 'condition' in fields:
                    rec['condition'] = random.choice(['New', 'Used'])
            # Attach other conditional fields
            if 'credit_band' in fields:
                rec['credit_band'] = random.choice(['Excellent', 'Good', 'Fair', 'Poor'])
            if 'dealer_id' in fields:
                rec['dealer_id'] = f'D{random.randint(1,50):03d}'
            if 'zipcode' in fields:
                rec['zipcode'] = str(random.randint(10000, 99999))
            if 'vin' in fields:
                rec['vin'] = f'VIN{random.randint(1000000, 9999999)}'
            if 'offer_amount' in fields:
                rec['offer_amount'] = random.randint(10000, 70000)
            # Attach weight
            rec['weight'] = weights.get(ev, 0)
            records.append(rec)
            timestamp += timedelta(minutes=random.randint(1, 10))
            i += 1

pd.DataFrame(records).to_csv(OUTPUT_PATH, index=False)
print(f'Generated {len(records)} events to {OUTPUT_PATH}')

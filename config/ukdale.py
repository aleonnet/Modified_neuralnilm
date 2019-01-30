""" Config for the III dataset """


# Windows (training, validation, testing)
WINDOWS = {
    'train': {
        1: ("2014-01-01", "2015-12-31"),
        2: ("2014-01-01", "2015-12-31"),
        3: ("2014-01-01", "2015-12-31"),
        4: ("2014-01-01", "2015-12-31"),
        5: ("2015-01-01", "2015-12-31")
    },
    'unseen_activations_of_seen_appliances': {
        1: ("2015-10-01", "2015-12-31"),
        2: ("2015-10-01", "2015-12-31"),
        3: ("2015-10-01", "2015-12-31"),
        4: ("2015-10-01", "2015-12-31"),
        5: ("2014-01-01", "2014-10-31")
    },
    'unseen_appliances': {
        1: ("2015-05-01", "2015-08-31"),
        2: ("2015-05-01", "2015-08-31"),
        3: ("2015-05-01", "2015-08-31"),
        4: ("2015-05-01", "2015-08-31"),
        5: ("2014-01-01", "2014-11-30")
    }
}
# Appliances
APPLIANCES = [
    'kettle',
    'microwave',
    'dish washer',
    'washing machine',
    'fridge',
]


# Training & validation buildings, and testing buildings
BUILDINGS = {
    'kettle': {
        'train_buildings': [ 1,2],
        'unseen_buildings': [5],
    },
#    'microwave': {
#        'train_buildings': [ 1,3,4],
#        'unseen_buildings': [2, 5],
#    },
#    'washing machine': {
#        'train_buildings': [ 1],
#        'unseen_buildings': [5],
#    },
    'dish washer': {
        'train_buildings': [ 1,2],
        'unseen_buildings': [5],
    },
#    'fridge': {
#        'train_buildings': [ 1,3,4],
#        'unseen_buildings': [2, 5],
#    },
}

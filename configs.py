
DPATH = '/home/ubuntu/datasets-aux/pollution/'

savedir = 'datasets/'

SEGMENTS = [
    {
        'locations': [
            ('Priti Sood Sayeed', 'E47A'),
            ('Arun Duggal', '20CA'),
            ('U.S. Embassy', 'CBC7'),
            ('Segel Design', 'BC46'),
            ('Jamun', 'BB4A'),
            ('Vihara', '91B8'),
            ('Prateek Mittal', '498F'),
            ('Nischal', 'C0A7'),
            ('Nita', '5D7A'),
        ],
        'start': '05/5/2018',
        'end': '05/20/2018',
    },
    {
        'locations': [
            ('Kailas Office', 'A9BE'),
            ('Parvathi', 'E8E4'),
            ('U.S. Embassy', 'CBC7'),
            ('Shubhra Mittal', '113E'),
            ('Smart air', '2E9C'),
            ('Lejo', '72CA'),
            ('Nita', '5D7A'),
        ],
        'start': '08/15/2018',
        'end': '08/28/2018',
    },
    {
        'locations': [
            ('U.S. Embassy', 'CBC7'),
            ('Senjuti Banerjee', '4BE7'),
            # ('Roy Imaging', '8E2A'),
            ('Vihara', '91B8'),
            ('Sujoy Home', '1FD7'),
            ('Prateek Mittal', '498F'),
            ('Nischal', 'C0A7'),
            ('Lejo', '72CA'),
            ('Nita', '5D7A'),
        ],
        'start': '07/10/2018',
        'end': '07/24/2018',
    },
]


EXCLUDE = {
    0: [
        'Sirifort, New Delhi - CPCB',
        'Dr. Karni Singh Shooting Range, Delhi - DPCC',
        'Sri Aurobindo Marg, Delhi - DPCC',
    ],
    1: [
        'Sirifort, New Delhi - CPCB',
        'Lodhi Road, New Delhi - IMD',
        'ITO, New Delhi - CPCB',
        'Punjabi Bagh, Delhi - DPCC',
    ],
    # 2: [
    #     'Sirifort, New Delhi - CPCB',
    #     'Punjabi Bagh, Delhi - DPCC',

    #     # 'Lodhi Road, New Delhi - IMD',
    #     # 'Jawaharlal Nehru Stadium, Delhi - DPCC',
    #     # 'Lodhi Road, New Delhi - IMD',
    #     # 'Sri Aurobindo Marg, Delhi - DPCC',
    # ]
}

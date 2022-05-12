import numpy as np
import random
# from numpy.random import seed
# seed(2021)

def random_neg(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

# def neg_neighbor(neighbor_dict, itemid):
#     neighor_set = neighbor_dict[itemid-1]
#     t = random.choice(neighor_set)
#     while t == itemid-1:
#         t = random.choice(neighor_set)
#     return t
def bucketized(seconds):
    boundaries = list(range(0, 11))
    time = np.log2(seconds+1)
    return np.searchsorted(boundaries, time)

class Sampler(object):
    def __init__(self, len_dict, session_dict, session_time_dict=None, neighbor_dict=None, item_dict=None, neg_num=None, batch_size=1024):
        print('Sampler init begin...')
        self.session_num = len(session_dict)
        self.batch_size = batch_size
        self.batch_num = 0
        self.batch_i = 0
        self.neighbor_dict = neighbor_dict
        self.item_dict = item_dict
        if item_dict!=None:
            self.item_num = len(item_dict)
        self.neg_num = neg_num
        self.len_dict = len_dict
        self.session_dict = session_dict
        self.session_time_dict = session_time_dict
        self.session_id_batches = []

        for slen, session_ids in self.len_dict.items():
            random.shuffle(session_ids)
            while(len(session_ids)>batch_size):
                self.session_id_batches.append(session_ids[:batch_size])
                self.batch_num += 1
                session_ids = session_ids[batch_size:]
            if len(session_ids):
                self.session_id_batches.append(session_ids)
                self.batch_num += 1
        random.shuffle(self.session_id_batches)
        print('Sampler init finished, batch size : {}, # batch: {}.'.format(self.batch_size, self.batch_num))

    def next_batch(self):
        batch_in = []
        batch_out = []
        batch_publisht_month = []
        batch_publisht_week = []
        batch_publisht_hour = []
        batch_publisht_day = []
        batch_publisht_minute = []
        batch_clickt_week = []
        batch_clickt_hour = []
        batch_clickt_day = []
        batch_clickt_month = []
        batch_clickt_minute = []
        batch_neg = []
        click_gap = []
        for session_id in self.session_id_batches[self.batch_i]:
            batch_in.append(self.session_dict[session_id][:-1])
            batch_out.append(self.session_dict[session_id][-1]-1)
            neg = []
            gap = []
            if self.session_time_dict:
                days = []
                weeks = []
                hours = []
                minutes = []
                months = []
                isotime_l = None
                for t in self.session_time_dict[session_id][:-1]:
                    isotime = t['publish_t']
                    months.append(isotime.month)
                    days.append(isotime.day)
                    weeks.append(isotime.isoweekday())
                    hours.append(isotime.hour+1)
                    minutes.append(isotime.minute+1)
                    isotime_c = t['click_t']
                    gap.append(bucketized(t['active_t'])) # if use Adressa dataset 

                    # if use Gloco dataset:

                #     if isotime_l!=None:
                #         gap.append(bucketized((isotime_c-isotime_l).seconds))
                #     isotime_l = isotime_c
                # gap.append(bucketized((self.session_time_dict[session_id][-1]['click_t']-isotime_l).seconds))
                if self.neighbor_dict:
                    # neg = self.neg_neighbor_from_impre(int(session_id.split('_')[0]))
                    # neg = self.neg_neighbor(self.session_dict[session_id][-1]-1)
                    while len(neg)<self.neg_num:
                        neg.append(np.random.randint(0, self.item_num))
                batch_publisht_month.append(months)
                batch_publisht_day.append(days)
                batch_publisht_week.append(weeks)
                batch_publisht_hour.append(hours)
                batch_publisht_minute.append(minutes)
                batch_clickt_month.append(isotime_c.month-1)
                batch_clickt_week.append(isotime_c.isoweekday()-1)
                batch_clickt_hour.append(isotime_c.hour)
                batch_clickt_day.append(isotime_c.day-1)
                batch_clickt_minute.append(isotime_c.minute)
            batch_neg.append(neg)
            click_gap.append(gap)
        self.batch_i += 1
        return batch_in, batch_out, (batch_publisht_month, batch_publisht_day, batch_publisht_week, batch_publisht_hour, batch_publisht_minute), (batch_clickt_month, batch_clickt_day, batch_clickt_week, batch_clickt_hour, batch_clickt_minute), batch_neg, click_gap

    def has_next(self):
        return self.batch_i<self.batch_num

    def neg_neighbor_from_impre(self, sessionid):
        neighor_set = self.neighbor_dict[sessionid]
        neg = []
        cnt = 0
        while len(neg)<self.neg_num:
            cnt += 1
            randomid = random.choice(neighor_set)
            if randomid in self.item_dict:
                neg.append(self.item_dict[randomid]-1)
            if cnt>20:
                break
        while len(neg)<self.neg_num:
            neg.append(np.random.randint(0, self.item_num))
        return neg
    
    def neg_neighbor(self, itemid):
        neighor_set = self.neighbor_dict[itemid]
        neg = []
        while len(neg)<self.neg_num:
            randomid = random.choice(neighor_set)
            if randomid != itemid:
                neg.append(randomid)
        return neg
from typing import List, Tuple

def get_tsvalues(t: int, arrivaltimes: List[int]) -> Tuple[int, int]:
    '''return (time_since_event, time_to_event)'''
    tse_candidates = [t-v for v in arrivaltimes if t >= v]
    tte_candidates = [v-t for v in arrivaltimes if v > t]
    return (
        min(tse_candidates) if len(tse_candidates) > 0 else None, 
        min(tte_candidates) if len(tte_candidates) > 0 else None,
    )

def get_input_t(t: int, arrivaltimes: list, training: bool = False) -> List[List[list]]:
    '''
    return numpy array of shape
        (num_obs, num_evtypes, 4=[tse, tte, prefirst, postlast]) if training
        (num_obs, num_evtypes, 2=[tse, prefirst]) otherwise
    tse, tte are -1 if prefirst is 1, postlast is 1 respectively
    '''
    # len(input_t) == num_obs
    input_t = []
    for arrt_obs in arrivaltimes:
        # len(input_type) == num_evtypes
        input_type = []
        for arrt_type in arrt_obs:
            tse, tte = get_tsvalues(t, arrt_type)
            prefirst, postlast = tse is None, tte is None
            resl = [
                # tse, tte, prefirst, postlast
                -1 if prefirst else tse, 
                -1 if postlast else tte,
                 1 if prefirst else 0,
                 1 if postlast else 0,
            ] if not training else [
                # tse, prefirst
                -1 if prefirst else tse, 
                 1 if prefirst else 0,
            ]
            input_type.append(resl)
        input_t.append(input_type)
    return input_t

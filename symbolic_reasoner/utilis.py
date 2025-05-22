from typing import List, Dict, Set, Iterable, Tuple, Any
from itertools import accumulate, permutations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import multiprocessing
from multiprocessing import Process, Queue



def count_not_none(lst):
    """
        Count the number of elements in the list that is not None
    """
    return sum(1 for item in lst if item is not None)



def number_to_polygon_name(n : int) -> str:
    number_to_name : Dict[int, str] = {
        3: 'Triangle',
        4: 'Quadrilateral',
        5: 'Pentagon',
        6: 'Hexagon',
        7: 'Heptagon',
        8: 'Octagon',
        9: 'Nonagon',
        10: 'Decagon'
    }
    return number_to_name.get(n, f'{n}-Polygon')


#------------------------------------------Iterables--------------------------------------------------
def cyclic_pairs(lis : Iterable[Any]) -> List[Tuple[Any, Any]]:
    if not isinstance(lis, list):
        lis = list(lis)
        
    return list(zip(lis, lis[1:] + lis[:1]))


#------------------------------------------List --------------------------------------------------
def list_intersection(*lists: Iterable[List[Any]]) -> List[Any]:
    return list(
        accumulate(
        lists,
        lambda x, y: [elem for elem in x if elem in y],
        initial = lists[0]
        )
    )[-1]

def list_union(*lists: Iterable[List]) -> List:
    
    return list(
        accumulate(
        lists,
        lambda x, y: x + [elem for elem in y if elem not in x],
        initial = []
        )
    )[-1]


# ------------------------------------------Mappings----------------------------------------------
def injection_mappingQ(mapping : dict) -> bool:
    '''
        Check if the mapping is an injection
    '''
    return len(set(mapping.values())) == len(mapping)

def consistent_mapping(mapping1 : dict, mapping2 : dict) -> bool:
    for key, value in mapping1.items():
        if key in mapping2 and mapping2[key] != value:
            return False
    return True

def consistent_mappings(*mappings : Iterable[dict] ) -> bool:
    for i in range(len(mappings)):
        for j in range(i+1, len(mappings)):
            if not consistent_mapping(mappings[i], mappings[j]):
                return False
    return True

def merge_mappings(*mappings : Iterable[dict] ) -> Dict:
    result = {}
    for mapping in mappings:
        result.update(mapping)
    return result


def inverse_mapping(mapping : dict) -> dict:
    return {value: key for key, value in mapping.items()}




#------------------------------------------Permutations----------------------------------------------


def cyclic_permutation(lis : List) -> List[List]:
    """ Return all cyclic permutations of the list """
    return [lis[i:] + lis[:i] for i in range(len(lis))]


def permutation_except_last(lis : List) -> List[List]:
    """ Permute the list except the last element """
    return list(map(lambda l: l + (lis[-1], ), permutations(lis[:-1])))

def alternating_group_permutations(lis: List) -> List[List]:
    """ Permute the list using the alternating group """
    return [lis[k:] + lis[:k] for k in range(len(lis))] + [lis[-k::-1] + lis[:-k:-1] for k in range(len(lis))]



#--------------------------------------------Singleton default dict-----------------------------------


class SingletonDefaultDict(defaultdict):
    """
        A class to create a singleton default dict
        the default value of this dict is a singleton set
    """
    def __init__(self):
        super().__init__(set)

    def __missing__(self, key):
        self[key] = {key}
        return self[key]
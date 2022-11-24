import numpy as np

def get_vector_str(x : np.ndarray):
    max_len = len(str(x.max()))
    extreme = " — " + " "*max_len + "—\n"
    res = "\n" + extreme
    for value in x:
        number_str = str(value)
        number_in_vec_str = "| " + number_str + " "*(max_len-len(number_str)) + "  |\n"
        res = res + number_in_vec_str
    res = res + extreme
    return res

def get_vector_str_from_list(L : list[np.ndarray], ncol : int):
    
    max_len_list = [len(str(x.max())) for x in L]
    extreme_list = [" — " + " "*max_len + "—" for max_len in max_len_list]
    res = "\n" + extreme
    for x in L :
        for value in x:
            number_str = str(value)
            number_in_vec_str = "| " + number_str + " "*(max_len-len(number_str)) + "  |"
            res = res + number_in_vec_str
    
    
    return res

if __name__ == "__main__":
    x = np.array([
        1,
        2,
        3,
        35,
        897,
        1,
        2,
        1086,
        1,
        1987
    ])
    
    print(get_vector_str(x))

import numpy as np

def main():
    import cuda_python
    
    lerp = cuda_python.common.execute_lerp
    
    # Tabulated values
    table = np.arange(0, 5, 0.1, dtype=np.float32)
    
    
    query = np.random.rand(5) * len(table)
    query = query.astype(np.float32)
    query = query.tolist()
    
    ret = lerp(query, table.tolist(), 1.0)
    
    assert len(ret) == len(query)
    
    for q, r in zip(query, ret):
        print('Q:', (q/10.0), 'R:', r)
        assert np.abs(r - (q/10.0)) < 1e-3
    
    print(query)
    print(ret)



if __name__ == '__main__':
    main()
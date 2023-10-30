import pickle
import random

def main():
    with open('color_all_data.pkl', 'r') as f:
        all_data = pickle.load(f)

    uniqe_data = []
    
    for ele in all_data:
        if ele not in uniqe_data:
            uniqe_data.append(ele)

    with open('color_unique_data.pkl', 'w') as f:
        pickle.dump(uniqe_data, f)

    class_data = {}
    
    # print(uniqe_data)

    for ele in uniqe_data:
        if ele[1] not in class_data.keys():
            # print(uniqe_data[1])
            class_data[ele[1]] = []
        
        class_data[ele[1]].append(ele)

    
    with open('color_class_data.pkl', 'w') as f:
        pickle.dump(class_data, f)

    balanced_data = []
    sample_number = 500
    for k in class_data.keys():
        sampled_data = random.sample(class_data[k], sample_number)
        for ele in sampled_data:
            balanced_data.append([ele[0], int(ele[1])])

    with open('color_balanced_data.pkl', 'w') as f:
        pickle.dump(balanced_data, f)


if __name__ == '__main__':
    main()
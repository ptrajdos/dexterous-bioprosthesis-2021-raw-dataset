import arff
import pandas as pd
import os

def save_arff(X,y,t,arff_path,relation_name=None):
    """
    Save dataset as an arff file
    Arguments:
    ---------
    X -- data representation objects x attributes. numpy array or pandas dataframe
    y -- object class
    t -- timestamp (numeric value) or None
    arff_path -- path to the created arff file
    relation_name -- name of the relation or None

    """

    if relation_name is None:
        int_rel_name = os.path.splitext( os.path.basename(arff_path) )[0]
    else:
        int_rel_name = relation_name
    
    if not isinstance(X,pd.DataFrame):
        att_names = [ "A{}".format(i) for i in range( X.shape[1] ) ]
        Xe = pd.DataFrame(X,columns=att_names)
    else:
        Xe = X

    attribute_types = [(str(c),'NUMERIC') for c in Xe.columns.values]
    str_y = [str(yy) for yy in y]
    
    if t is not None:
        timestamp_series = pd.Series(t,)
        timestamp_attrib_type = [ ('timestamp','NUMERIC') ]
        Xe['timestamp'] = timestamp_series
        attribute_types+=timestamp_attrib_type


    class_attribute = [('class',sorted(list(set(str_y))))]
    attribute_types+= class_attribute
    class_names_series = pd.Series(str_y, dtype='category')
    Xe['class'] = class_names_series

    arff_dict= {
        'attributes':attribute_types,
        'data': Xe.values,
        'relation':int_rel_name,
        'description':''
    }

    with open(arff_path, "w", encoding="utf8") as f:
            arff.dump(arff_dict,f)

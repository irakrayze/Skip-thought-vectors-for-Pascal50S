import skipthoughts
import scipy.io
import theano
import numpy

"""
This method recieves as input Pascal50S data set from http://ramakrishnavedantam928.github.io/cider/
It returns 2 dictionaries: the first one is Images_Dictionary,
a dictionary with image ids and the corresponding 50 descriptions for each image;
the second one is Image_vectors_dic, 
a dictionary with image ids and skip-thought vectors for each description following https://github.com/ryankiros/skip-thoughts
"""
def Extract_Vector(pascal_file):
    #pascal_file='/home/ira/Documents/pascal50S.mat'
    
    # extracting matlab struct
    mat = scipy.io.loadmat(pascal_file, struct_as_record=True)
    images_num=mat['train_sent_final'][0].size #1000 images

    Images_Dictionary={} # the key is the image id , the value is the list of sentences per each image
    for image in range(0,images_num):
        image_id="%04d" % (image,)
        image_obj=mat['train_sent_final'][0,image_id]
        image_url=image_obj[0]
        url=numpy.array(image_url).tolist()
        url_str_uni = ''.join(url)
        url_str_utf=url_str_uni.encode('utf-8')
        image_name=url_str_utf[url_str_utf.rfind('_')+1:url_str_utf.rfind('.')] #extract the name of the image from the url
        image_sentences=image_obj[1][0]
        num_sentences=image_sentences.size
        Sentences_List=[]
        for sent in range(0,num_sentences):
            sent=numpy.array(image_sentences[sent]).tolist()
            sent_str_uni = ''.join(sent)
            sent_str_utf = sent_str_uni.encode('utf-8') # one sentence as a string
            Sentences_List.append(sent_str_utf)
        if image_name in Images_Dictionary:
            Images_Dictionary[image_name].append(Sentences_List)
        else:
            Images_Dictionary[image_name]=Sentences_List

    # Generating skip-thought vectors following https://github.com/ryankiros/skip-thoughts
    Image_vectors_dic={}
    model = skipthoughts.load_model()
    for key in Images_Dictionary:
        Sentences=Images_Dictionary[key]
        vectors=skipthoughts.encode(model, Sentences)
        if key in Image_vectors_dic:
            Image_vectors_dic[key].append(vectors)
        else:
            Image_vectors_dic[key]=vectors

    return Images_Dictionary, Image_vectors_dic, num_sentences

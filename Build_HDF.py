import h5py
import scipy.io as scio
import Extract_Vector.py

"""
This method recieves 3 inputs: docAll- a txt file with the names(ids) of all images in Pascal50S, 
docVis1 - a mat file with visual features for each of the images, 
docVis2 - a mat file with different visual features for each of the images.
It stores all the data: image id, the 50  relevant descriptions per image,  the image visual features and the description 
vectors in a convinient fromat- HDF https://www.hdfgroup.org/HDF5/whatishdf5.html
"""
def Build_HDF(docAll, docVis1, docVis2 ):

    dataset_name = 'Pascal50S'

    splits = ['train' ] #in test there is only one description in each language
    h5output = h5py.File(dataset_name+".h5", "w")

    for spl in splits:
        split = h5output.create_group(spl)

        ########### The visual features order is according to the order in docAll   #############
        #docAll='Image_Names_Pascal.txt'
        All_Dict = {}
        with open(docAll, 'r') as f:
            for cnt, line in enumerate(f):
                line=line.strip()
                dot=line.rfind(".")
                redun=line.rfind("_")
                image_id=line[redun+1:dot]
                All_Dict[image_id] = cnt #the value is the image id and the key is the count, count starts with 0

        #docVis1='Pascal_vgg19_relu7_feats.mat' #file with image vecotrs dimension=4096

        #docVis2='Pascal_vgg19_conv54_feats.mat'

        data1=scio.loadmat(docVis1)
        data2=scio.loadmat(docVis2)

        vis1=data1.get('feats') #shape=(4096,1000)
        vis2=data2.get('feats') #shape=(135000,1000)

        #################    ALL SENTENCES FOR THE IMAGES    ############################################################################
        pascal_file = '/home/ira/Documents/pascal50S.mat'
        Images_Dictionary={}
        Image_vectors_dic={}
        Images_Dictionary, Image_vectors_dic, num_sentences=Extraxt_Sentence_Vector(pascal_file) #num_sentences=50
        
        ########################################  BUILD DATASET #######################################################
        dt = h5py.special_dtype(vlen=unicode)

        for count, im in enumerate(All_Dict.keys()):
            container = split.create_group("%06d" % count)  #creates sequtioal ids
            dset_id = container.create_dataset("img_id", (1,), dtype=dt)
            dset_edis = container.create_dataset("eng_dis", (num_sentences,), dtype=dt) #for each image we have in pascal 50s 50 descriptions
            dset_sent_vec=container.create_dataset("sent_vec", (num_sentences,vis1.shape[0]), dtype='float32') # the vectors representing the sentences
            dset_vis1 = container.create_dataset("vis_feats1", (vis1.shape[0],), dtype='float32') #the dimension of the vector of each image ---change to auto
            dset_vis2 = container.create_dataset("vis_feats2", (vis2.shape[0],), dtype='float32')
            dset_id[:] = unicode(im) #insert the id number to the dataset
            ################### Visual Feats  ##################
            dset_vis1[:]=vis1[:,All_Dict[im]]
            dset_vis2[:]=vis2[:,All_Dict[im]]
            #################### English Dis  #######################
            Sen_list=Images_Dictionary[im]
            for part in range(0,num_sentences):
                dset_edis[part]=Sen_list[part]
            ############# Sentences Vectors #########################
            Sen_Vec=Image_vectors_dic[im]
            for r in range (0,num_sentences):
                dset_sent_vec[r]=Sen_Vec[r,:]
 return

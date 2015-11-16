import numpy as np
import matplotlib.pyplot as plt

def displayData(images):
    ''' Display image samples in images in a nice 2D grid
    ''' 

    # Set example_width automatically if not passed in
    # TODO , add capability to pass in the width
    # if ~exist('example_width', 'var') || isempty(example_width) 
    #	example_width = round(sqrt(size(X, 2)));


    # Set the width
    example_width = np.round(np.sqrt(images.shape[1]))
 
    # Gray Image
    # colormap(gray);
    plt.gray()
    
    # get the size of the images array
    m_images, n_images = images.shape
    example_height = (n_images / example_width)   

    # Compute number of items to display
    display_rows = np.floor(np.sqrt(m_images))
    display_cols = np.ceil(m_images / display_rows)

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                     pad + display_cols * (example_width + pad)))

    curr_ex = 0
    for j in np.arange(display_rows):
        for i in np.arange(display_cols):
            if curr_ex >= m_images: 
                break 
            # Copy the patch
            max_val = max(abs(images[curr_ex, :]))
            row = pad + j * (example_height + pad) + np.arange(example_height)
            row = row.astype("int")
            col = pad + i * (example_width + pad) + np.arange(example_width)
            col = col.astype("int") 
            display_array[np.ix_(row,col)]  = images[curr_ex, :].reshape(
                  (example_height, example_width)) / max_val
            curr_ex = curr_ex + 1
        if curr_ex >= m_images: 
            break 

    # Display the image
    plt.ion()
    h = plt.imshow(display_array)
    h.axes.get_yaxis().set_visible(False)
    h.axes.get_xaxis().set_visible(False)
    plt.gray()
    plt.draw()

    return h
	

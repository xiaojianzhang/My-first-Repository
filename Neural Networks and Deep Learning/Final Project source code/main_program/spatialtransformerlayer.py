import theano.tensor as T
import lasagne
import theano

class SpatialTransformerLayer(lasagne.layers.MergeLayer):
    """
    functions in lasagne class:
    
    __init__
    
    get_output_shape_for
    
    get_output_for
    
    """
    def __init__(self, incoming, ds_rate=1.0, **kwargs):
        super(SpatialTransformerLayer, self).__init__(incoming, **kwargs)
        self.ds_rate=ds_rate
    
    
    def get_output_shape_for(self, input_shapes):
        img_shape,xfm_shape = input_shapes
        n_batch,n_channel,i_height,i_width = img_shape

        o_height = int(i_height//self.ds_rate)
        o_width = int(i_width//self.ds_rate)
        
        output_shapes = n_batch,n_channel,o_height,o_width
        return list(output_shapes)
    
    
    def get_output_for(self, inputs, **kwargs):
        img,xfm = inputs

        n_batch,n_channel,i_height,i_width = img.shape

        o_height = ((i_height//self.ds_rate)).astype('int64')
        o_width =  ((i_width//self.ds_rate)).astype('int64')
        
        # grid generater        
        target_grid = grid_generater(o_height,o_width)

        # localization net
        # xfm = localization_net()

        # sampler
        source_grid = sampler(xfm,target_grid)
        
        output = apply_sampling(source_grid,img,o_height,o_width)
        return output

def grid_generater(o_height,o_width):
    """
    Generate a regular grid as the target coordinates
    return two flattened vectors similar to the behavior of meshgrid in MATLAB
    """
    # default flatten() behavior is C-style row-major row y col x
    target_grid_col = (T.arange(o_width).repeat(o_height).reshape((o_width,o_height)).transpose()).astype(theano.config.floatX)
    target_grid_row = T.arange(o_height).repeat(o_width).reshape((o_height,o_width)).astype(theano.config.floatX)

    # T.mgrid doesn't work for tensor variable
    # target_grid = T.mgrid[T.arange(o_height),T.arange(o_width)]
    
    # normalize to (-1,1)
    # 0 1 2 0 1 2
    target_grid_col = target_grid_col.flatten()/(o_width-1.0)*2.0-1.0
    # 0 0 0 1 1 1
    target_grid_row = target_grid_row.flatten()/(o_height-1.0)*2.0-1.0
    return [target_grid_col,target_grid_row]


def localization_net():
    """
    localization net
    """

def sampler(xfm,target_grid):
    # affine transform
    # n_batch x 2 x 3
    
    xfm = T.reshape(xfm,(xfm.shape[0],2,3))
    # n_batch x 2 x (heightxwidth)
    source_grid = T.dot(xfm,
        T.stack([target_grid[0],target_grid[1],T.ones_like(target_grid[0])]))
    source_grid = T.clip(source_grid,-1.0,1.0)
    
    return source_grid

def apply_sampling(source_grid,img,o_height,o_width):
    n_batch,n_channel,i_height,i_width = img.shape
#    # 0 0 0 1 1 1 2 2 2 0 0 0 1 1 1 2 2 2
#    source_grid_row = source_grid[:,0,:].reshape((n_batch,1,-1)).repeat(n_channel,axis=1).flatten()
#    # 0 1 2 0 1 2 0 1 2
#    source_grid_col = source_grid[:,1,:].reshape((n_batch,1,-1)).repeat(n_channel,axis=1).flatten()
    
    source_grid_row = source_grid[:,1,:].flatten()    
    source_grid_col = source_grid[:,0,:].flatten()
    
    
    # (T.dot(T.ones((2,2,3)),T.stack([T.ones((10,1)).flatten(),T.ones((10,1)).flatten(),T.ones((10,1)).flatten()]))[:,0,:]*2
    # + T.dot(T.ones((2,2,3)),T.stack([T.ones((10,1)).flatten(),T.ones((10,1)).flatten(),T.ones((10,1)).flatten()]))[:,1,:]).eval().shape
    # batch1 row1 row2 row3 ... batch2 row1 ...
    img = img.dimshuffle((0,2,3,1)).reshape((-1,n_channel)).astype(theano.config.floatX)
    source_grid_row=(source_grid_row+1.0)/2.0*(i_height-1.0)
    source_grid_col=(source_grid_col+1.0)/2.0*(i_width-1.0)
    source_grid_row_floor=T.floor(source_grid_row).astype('int64')
    source_grid_col_floor=T.floor(source_grid_col).astype('int64')
    source_grid_row_ceil=T.clip(source_grid_row_floor+1,0,i_height-1).astype('int64')
    source_grid_col_ceil=T.clip(source_grid_col_floor+1,0,i_width-1).astype('int64')
    # output = img[source_grid_row*i_width+source_grid_col].reshape((n_batch,n_channel,o_width,o_height))
    
    batch_base=(T.arange(n_batch)*(i_height*i_width)).repeat(o_height*o_width).astype('int64')
    
    # bilinear interpolation
    output_nw = img[source_grid_row_floor*i_width+source_grid_col_floor+batch_base,:]
    output_ne = img[source_grid_row_floor*i_width+source_grid_col_ceil+batch_base,:]
    output_sw = img[source_grid_row_ceil*i_width+source_grid_col_floor+batch_base,:]
    output_se = img[source_grid_row_ceil*i_width+source_grid_col_ceil+batch_base,:]
    
    weight_nw = ((source_grid_row_ceil-source_grid_row)*(source_grid_col_ceil-source_grid_col)).reshape((-1,1)).repeat(n_channel,axis=1)
    weight_ne = -((source_grid_row_ceil-source_grid_row)*(source_grid_col_floor-source_grid_col)).reshape((-1,1)).repeat(n_channel,axis=1)
    weight_sw = -((source_grid_row_floor-source_grid_row)*(source_grid_col_ceil-source_grid_col)).reshape((-1,1)).repeat(n_channel,axis=1)
    weight_se = ((source_grid_row_floor-source_grid_row)*(source_grid_col_floor-source_grid_col)).reshape((-1,1)).repeat(n_channel,axis=1)
    
    output =(output_nw*weight_nw+output_ne*weight_ne+output_sw*weight_sw+output_se*weight_se)
    output = T.reshape(output,(n_batch,o_height,o_width,n_channel)).dimshuffle((0,3,1,2))
        
        
#    source_grid_row=((source_grid_row+1.0)/2.0*(i_height-1.0)).astype('int16')
#    source_grid_col=((source_grid_col+1.0)/2.0*(i_width-1.0)).astype('int16')
#    output = img[source_grid_row*i_width+source_grid_col].reshape((n_batch,n_channel,o_height,o_width))
    return output.astype(theano.config.floatX)
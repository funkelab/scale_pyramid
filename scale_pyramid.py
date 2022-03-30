import argparse
import daisy
import numpy as np
import skimage.measure
import zarr

# monkey-patch os.mkdirs, due to bug in zarr
import os
prev_makedirs = os.makedirs


def makedirs(name, mode=0o777, exist_ok=False):
    # always ok if exists
    return prev_makedirs(name, mode, exist_ok=True)


os.makedirs = makedirs


def downscale_block(in_array, out_array, factor, block):

    dims = len(factor)
    in_data = in_array.to_ndarray(block.read_roi, fill_value=0)

    in_shape = daisy.Coordinate(in_data.shape[-dims:])
    assert in_shape.is_multiple_of(factor)

    n_channels = len(in_data.shape) - dims
    if n_channels >= 1:
        factor = (1,)*n_channels + factor

    if in_data.dtype == np.uint64:
        slices = tuple(slice(k//2, None, k) for k in factor)
        out_data = in_data[slices]
    else:
        out_data = skimage.measure.block_reduce(in_data, factor, np.mean)

    try:
        out_array[block.write_roi] = out_data
    except Exception:
        print("Failed to write to %s" % block.write_roi)
        raise

    return 0


def downscale(in_array, out_array, factor, write_size):

    print("Downsampling by factor %s" % (factor,))

    dims = in_array.roi.dims()
    block_roi = daisy.Roi((0,)*dims, write_size)

    print("Processing ROI %s with blocks %s" % (out_array.roi, block_roi))

    downscale_task=daisy.Task(
        'downscale',
        out_array.roi,
        block_roi,
        block_roi,
        process_function=lambda b: downscale_block(
            in_array,
            out_array,
            factor,
            b),
        read_write_conflict=False,
        num_workers=60,
        max_retries=0,
        fit='shrink')
    
    done = daisy.run_blockwise([downscale_task])

    if not done:
        raise RuntimeError("daisy.Task failed for (at least) one block")

def create_scale_pyramid(in_file, in_ds_name, scales, chunk_shape, compressor={'id': 'zlib', 'level': 5}):

    ds = zarr.open(in_file)

    # make sure in_ds_name points to a dataset
    try:
        daisy.open_ds(in_file, in_ds_name)
    except Exception:
        raise RuntimeError("%s does not seem to be a dataset" % in_ds_name)

    if not in_ds_name.endswith('/s0'):

        ds_name = in_ds_name + '/s0'

        print("Moving %s to %s" % (in_ds_name, ds_name))
        ds.store.rename(in_ds_name, in_ds_name + '__tmp')
        ds.store.rename(in_ds_name + '__tmp', ds_name)

    else:

        ds_name = in_ds_name
        in_ds_name = in_ds_name[:-3]

    print("Scaling %s by a factor of %s" % (in_file, scales))

    prev_array = daisy.open_ds(in_file, ds_name)

    if chunk_shape is not None:
        chunk_shape = daisy.Coordinate(chunk_shape)
    else:
        chunk_shape = daisy.Coordinate(prev_array.data.chunks)
        print("Reusing chunk shape of %s for new datasets" % (chunk_shape,))

    if prev_array.n_channel_dims == 0:
        num_channels = None
    elif prev_array.n_channel_dims == 1:
        num_channels = prev_array.shape[0]
    else:
        raise RuntimeError(
            "more than one channel not yet implemented, sorry...")

    for scale_num, scale in enumerate(scales):

        try:
            scale = daisy.Coordinate(scale)
        except Exception:
            scale = daisy.Coordinate((scale,)*chunk_shape.dims())

        next_voxel_size = prev_array.voxel_size*scale
        next_total_roi = prev_array.roi.snap_to_grid(
            next_voxel_size,
            mode='grow')
        next_write_size = chunk_shape*next_voxel_size

        print("Next voxel size: %s" % (next_voxel_size,))
        print("Next total ROI: %s" % next_total_roi)
        print("Next chunk size: %s" % (next_write_size,))

        next_ds_name = in_ds_name + '/s' + str(scale_num + 1)
        print("Preparing %s" % (next_ds_name,))

        next_array = daisy.prepare_ds(
            in_file,
            next_ds_name,
            total_roi=next_total_roi,
            voxel_size=next_voxel_size,
            write_size=next_write_size,
            dtype=prev_array.dtype,
            num_channels=num_channels
            compressor=compressor)

        downscale(prev_array, next_array, scale, next_write_size)

        prev_array = next_array


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create a scale pyramide for a zarr/N5 container.")

    parser.add_argument(
        '--file',
        '-f',
        type=str,
        help="The input container")
    parser.add_argument(
        '--ds',
        '-d',
        type=str,
        help="The name of the dataset")
    parser.add_argument(
        '--scales',
        '-s',
        nargs='*',
        type=int,
        required=True,
        help="The downscaling factor between scales")
    parser.add_argument(
        '--chunk_shape',
        '-c',
        nargs='*',
        type=int,
        default=None,
        help="The size of a chunk in voxels")
    parser.add_argument(
        '--compressor',
        '-C',
        nargs='*',
        type=dict,
        default={'id': 'zlib', 'level': 5},
        help="The compressor as a dict where 'id' is compressor name and 'level' is compression level")

    args = parser.parse_args()

    create_scale_pyramid(args.file, args.ds, args.scales, args.chunk_shape)

import glob as glob                                            # grabbing file names
import pandas as pd                                            # making dataframe for exporting parameters
import numpy as np                                             # basic math
import skimage.io as io                                        # reading in images
import os
import trackpy as tp             # particle tracking toolbox
import matplotlib.pyplot as plt  # for plotting everything
from matplotlib import cm, colors
import seaborn as sns
import matplotlib.patches as mpatches
import cv2                                                     # for filtering vector fields
from skimage.feature import peak_local_max
import warnings
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
from image_plotting_tools import *
from interactive_plotting_tools import *
import czifile
import shutil

def calculate_length_and_velocity(x,y,frames, frame_duration=0.5):
    # Calculate delta_x and delta_y : distance traveled between consecutive frames
    delta_x = x[1:]-x[:-1]
    delta_y = y[1:]-y[:-1]
    # calculate the distance between points
    delta_step = np.sqrt(delta_x**2+delta_y**2)
    # path length is the sum of step sizes
    path_length = np.sum(delta_step)
    # calculate the length of time the path covers
    number_of_frames = len(x)
#     path_duration = (number_of_frames-1)*frame_duration
    # this accounts for skipped frames
    path_duration = np.sum(np.diff(frames) * frame_duration)
    # calculate the average velocity
    average_velocity = path_length/path_duration
    # calculate the total displacement
    delta_x_total = x[-1]-x[0]
    delta_y_total = y[-1]-y[0]
    actual_displacement = np.sqrt(delta_x_total**2+delta_y_total**2)
    # calculate the effective velocity
    effective_velocity = actual_displacement/path_duration
    
    return path_length, path_duration, average_velocity, actual_displacement, effective_velocity

def calculate_costheta(x, y):
    # hide error for dividing by zero since we know it's coming
    np.seterr(divide='ignore', invalid='ignore')
    # calculate the change between points
    dx = np.diff(x)
    dy = np.diff(y)
    # combine into an array for easy calculation
    v = np.vstack((dx, dy)).T
    # dot product divided by the magnitudes gives cos theta
    costheta = np.sum(v[:-1] * v[1:], axis=1) / np.sqrt((np.sum(v[:-1]**2, axis=1) * np.sum(v[1:]**2, axis=1)))
    theta = np.arccos(costheta)
    return theta

def make_parameter_textfile(filename, max_projection, frame_interval, um_per_pixel, filtering_kernel_size, gaussian_std,
                            min_peak_inten, min_peak_distance, feature_size, max_feature_displacement,
                            min_track_length_frames, min_track_length_nm, number_of_skipped_frames, use_peak_image, 
                            smoothing_kernel, anterograde_color, retrograde_color):
    # figure out filefolder
    if filename.find('/') == -1:
        folder = ''
    else:
        folder = filename[:filename.rfind('/')] + '/'
    
    # make a dictionary
    tracking_params = {
        'max_projection' : max_projection,
        'frame_interval' : frame_interval,
        'um_per_pixel' : um_per_pixel,
        'filtering_kernel_size' : filtering_kernel_size,
        'gaussian_std' : gaussian_std,
        'min_peak_inten' : min_peak_inten,
        'min_peak_distance' : min_peak_distance,
        'feature_size' : feature_size,
        'max_feature_displacement' : max_feature_displacement,
        'min_track_length_frames' : min_track_length_frames,
        'min_track_length_nm' : min_track_length_nm,
        'number_of_skipped_frames' : number_of_skipped_frames,
        'use_peak_image' : use_peak_image,
        'smoothing_kernel' : smoothing_kernel,
        'anterograde_color' : anterograde_color,
        'retrograde_color' : retrograde_color
    }
    
    #write the file
    pd.DataFrame.from_dict(data=tracking_params, orient='index').to_csv(folder + 'tracking_params.csv', header=False)
    
    return

def filter_image(imstack, filename, filtering_kernel_size, gaussian_std, save_image = True):
    # Make Laplace of Gaussian (LoG) filter
    # Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(filtering_kernel_size, gaussian_std)
    # Multiply the vector by it's transpose to make a matrix
    gaussian_kernel = gaussian_kernel * gaussian_kernel.T
    # Create a Laplacian
    Laplace = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
    # Convolve to make a 2D Laplace of Gaussian Filter
    LoG_kernel = cv2.filter2D(gaussian_kernel, -1, Laplace)

    # filter the image stack with a LoG filter
    imstack_LoG = np.zeros(imstack.shape)
    for i,plane in enumerate(imstack):
        imstack_LoG[i] = cv2.filter2D(plane.astype('float32'), cv2.CV_32F, LoG_kernel)

    # save the filtered image stack
    if save_image:
        io.imsave(filename[:-4] + '_filtered.tif', imstack_LoG.astype('float32'), check_contrast=False)
    
    return imstack_LoG

def find_peaks_in_image(imstack, filename, min_peak_distance, min_peak_inten, filtering_kernel_size = 15, gaussian_std = 2, save_image = True ):
    # create empty list to hold peak coordinates
    peak_list = []
    # create image stack to hold local peaks
    peak_stack = np.zeros(imstack.shape)
    # Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(filtering_kernel_size, gaussian_std)
    # Multiply the vector by it's transpose to make a matrix
    gaussian_kernel = gaussian_kernel * gaussian_kernel.T
    # find local peaks in each plane
    for i,plane in enumerate(imstack):
        # find local peaks
        plane_peaks = peak_local_max(plane, min_distance=min_peak_distance, threshold_abs=min_peak_inten)
        # save peak list
        peak_list.append(plane_peaks)
        # set peaks pixels to 1 in imagestack
        peak_stack[i][(plane_peaks[:,0],plane_peaks[:,1])] = 1
        # convolve with gaussian kernel to make peaks more visible in the iamge
        peak_stack[i] = cv2.filter2D(peak_stack[i].astype('float'), cv2.CV_32F, gaussian_kernel) * 100

    # save the peak image stack
    if save_image:
        io.imsave(filename[:-4] + '_peaks.tif', peak_stack.astype('float32'), check_contrast=False)

    return peak_stack, peak_list

def check_peaks_in_single_frame(imstack, peak_list, frame_num = 0, feature_size = 9, minmass = 50, separation = 3, max_inten = None, min_inten = None):
    # find features using trackpy
    potential_myosin = tp.locate(imstack[frame_num],feature_size, minmass=minmass, separation=3)

    
    # make figure
    fig, ax = plt.subplots(ncols=2)
    tp.annotate(potential_myosin, imstack[frame_num], ax=ax[0])
    ax[0].set_title('tracky intensity features')
    ax[0].axis('off')
    ax[1].imshow(imstack[frame_num], vmin = min_inten, vmax = max_inten)
    ax[1].plot(peak_list[frame_num][:,1], peak_list[frame_num][:,0],'.r')
    ax[1].set_title('peak finding approach')
    ax[1].axis('off')
    # show the image
    fig.show()
    return

def plot_track_overlays(myosin_tracks_filtered, filename, imstack_LoG, save_image=True):
    # Plot the trajectories overlaid on the the image
    myosin_traj_fig, myosin_traj_axis = plt.subplots()
    tp.plot_traj(myosin_tracks_filtered, superimpose=imstack_LoG[0], ax=myosin_traj_axis)
    myosin_traj_axis.set_title('Number of filtered tracks found : ' + str(myosin_tracks_filtered['particle'].nunique()))
    myosin_traj_fig.show()
    # save the tracked image
    if save_image:
        myosin_traj_fig.savefig(filename[:-4] + '_myosin_tracks_overlaid.png', dpi=300)
        
    return

def calculate_trajectory_parameters(myosin_tracks_filtered, filename, nm_per_pixel, frame_interval = 1, min_track_length_nm = 1000, smoothing_window_size = 5):

    switch_distance_threshold = 350 # in nm
    switch_frame_threshold = 5 

    # change the value set for index from the frame to the particle
    myosin_tracks_filtered_indexed = myosin_tracks_filtered.set_index('particle')

    # make a list of all the unique particle values
    unique_particle=np.unique(myosin_tracks_filtered['particle'].values)


    # create empty lists for all the variables we're going to calculate
    all_path_lengths, all_path_duration, all_particle_velocities, all_actual_displacement, all_effective_velocities = [], [], [], [], []
    all_costhetas, all_mean_costheta = [],[]
    all_tracks_x, all_tracks_y, all_tracks_frames = [],[],[]
    all_tracks_x_smoothed, all_tracks_y_smoothed, all_tracks_frames_smoothed = [],[],[]
    all_path_lengths_smoothed, all_path_duration_smoothed, all_particle_velocities_smoothed, all_actual_displacement_smoothed, all_effective_velocities_smoothed = [], [], [], [], []
    
    particle_list = []


    # loop through each unique particle
    for puncta in unique_particle:
        # make a table of the points for each individual myosin puncta
        individual_table = myosin_tracks_filtered_indexed.loc[puncta]

        # get the initial starting position
        x0=individual_table.iloc[0]['x']
        y0=individual_table.iloc[0]['y']
        #substract x0 and y0 from x and y
        x=individual_table['x']-x0
        x = x.values * nm_per_pixel
        y=individual_table['y']-y0
        y = y.values * nm_per_pixel
        
        # get the frames 
        track_frames = individual_table['frame'].values

        # calculate the trajectory parameters
        path_length,path_duration, particle_velocity, actual_displacement, effective_velocity = calculate_length_and_velocity(x, y, track_frames, frame_duration=frame_interval)

        # calculate the turning angles between each point
        costheta = calculate_costheta(x,y)
        
        #get smoothed trajectory points
        # Define your averaging kernel
        # window_size = smoothing_window_size
        kernel = np.ones((smoothing_window_size,))/smoothing_window_size

        # Convolve your linescan with your kernal
        # mode can equal 'full', 'same', 'valid'
        x_smoothed = np.convolve(individual_table['x'], kernel, mode='valid')
        x_smoothed = (x_smoothed - x_smoothed[0]) * nm_per_pixel
        y_smoothed = np.convolve(individual_table['y'], kernel, mode='valid')
        y_smoothed = (y_smoothed - y_smoothed[0]) *  nm_per_pixel
        track_frames_smoothed = track_frames[int(np.floor(smoothing_window_size/2)):-int(np.ceil(smoothing_window_size/2))+1]
         
        # calculate the trajectory parameters for smoothed traces
        path_length_smoothed,path_duration_smoothed, particle_velocity_smoothed, actual_displacement_smoothed, effective_velocity_smoothed = calculate_length_and_velocity(x_smoothed, y_smoothed, track_frames_smoothed, frame_duration=frame_interval)


        # add all our calculated parameters to their respective lists
        all_path_lengths.append(path_length)
        all_costhetas.append(costheta)
        all_mean_costheta.append(np.nanmean(costheta))
        all_particle_velocities.append(particle_velocity)
        all_actual_displacement.append(actual_displacement)
        all_effective_velocities.append(effective_velocity)
        all_path_duration.append(path_duration)
        all_tracks_x.append(x)
        all_tracks_y.append(y)
        all_tracks_frames.append(track_frames)
        all_tracks_x_smoothed.append(x_smoothed)
        all_tracks_y_smoothed.append(y_smoothed)
        all_tracks_frames_smoothed.append(track_frames_smoothed)
        all_path_lengths_smoothed.append(path_length_smoothed)
        all_path_duration_smoothed.append(path_duration_smoothed)
        all_particle_velocities_smoothed.append(particle_velocity_smoothed)
        all_actual_displacement_smoothed.append(actual_displacement_smoothed)
        all_effective_velocities_smoothed.append(effective_velocity_smoothed)
        particle_list.append(puncta)


    # save the dataframe with all the myosin track data to the folder
    myosin_tracks_filtered_indexed.to_hdf(filename[:-4] + '_tracks.h5', key = 'myosin', mode='w')
    # make a dictionary of the dataframe with the track data
    myosin_track_dict = {
        'particle' : particle_list,
        'pathlength' : all_path_lengths,
        'displacement' : all_actual_displacement,
        'actual_velocity' : all_particle_velocities,
        'effective_velocity' : all_effective_velocities,
        'costheta' : all_costhetas,
        'mean_costheta' : all_mean_costheta,
        'duration' : all_path_duration,
        'x' : all_tracks_x,
        'y' : all_tracks_y,
        'frames' : all_tracks_frames,
        'x_smoothed' : all_tracks_x_smoothed,
        'y_smoothed' : all_tracks_y_smoothed,
        'frames_smoothed' : all_tracks_frames_smoothed,
        'pathlength_smoothed' : all_path_lengths_smoothed,
        'displacement_smoothed' : all_actual_displacement_smoothed,
        'actual_velocity_smoothed' : all_particle_velocities_smoothed,
        'effective_velocity_smoothed' : all_effective_velocities_smoothed,
        'duration_smoothed' : all_path_duration_smoothed
    }

    # convert the dictionary to a dateframe
    myosin_trackdata_df = pd.DataFrame(myosin_track_dict)
    # sort the dataframe by tracklength
    myosin_trackdata_df.sort_values(['pathlength'], ascending=False)
    # save the dataframe to the folder
    myosin_trackdata_df.to_hdf(filename[:-4] + '_all_trackdata.h5', key = 'myosin', mode='w')

    myosin_trackdata_df = myosin_trackdata_df[myosin_trackdata_df['displacement'] > min_track_length_nm]
    myosin_trackdata_df.to_hdf(filename[:-4] + '_filtered_trackdata.h5', key = 'myosin', mode='w')
    # myosin_trackdata_df = myosin_trackdata_df.reset_index(drop=True)
    # myosin_trackdata_df['particle_id'] = myosin_trackdata_df.index.values

    # figure out the angle of flow from first and last points (use  the smoothed data to get a better approximation)
    flow_angle = []
    flow_direction = []
    flow_switch, flow_switch_position, flow_switch_travel = [],[],[]
    index_list = []
    for index, row in myosin_trackdata_df.iterrows():
        x = row['x_smoothed']
        y = -1 * row['y_smoothed']
        # calculate the angles of all the points
        angle = np.arctan2(y, x)
        # determine whether it switches directions
        distance_from_end = ((x-x[-1])**2 + (y-y[-1])**2) ** 0.5
        distance_max = np.where(distance_from_end == np.max(distance_from_end))
        distance_max_frame = distance_max[0][0]
        distance_max_travel = ((x[distance_max_frame])**2 + (y[distance_max_frame])**2) ** 0.5
        # check to see if it switches
        if (distance_max[0][0] > switch_frame_threshold) and (distance_max_travel > switch_distance_threshold):
            flow_switch.append('switches')
            flow_switch_position.append(distance_max[0][0])
            flow_switch_travel.append(distance_max_travel)
        else:
            flow_switch.append('straight')
            flow_switch_position.append(0)
            flow_switch_travel.append(0)
        # find the angle from the last point to determine whether it's retro or anterograde
        flow_angle.append(angle[-1])

    # calculate median angle of flow
    median_angle = np.median(flow_angle)
    x_median = np.cos(median_angle)
    y_median = np.sin(median_angle)

    # define retrograde and anterograde based on that median angle
    # for angle in flow_angle:
    #   if angle < (median_angle - np.pi/2) or angle > (median_angle + np.pi/2):
    #       flow_direction.append('anterograde')
    #   else:
    #       flow_direction.append('retrograde')

    for index, row in myosin_trackdata_df.iterrows():
        x = row['x_smoothed']
        y = -1 * row['y_smoothed']
        if (x_median*x[-1] + y_median*y[-1]) >= 0:
            flow_direction.append('retrograde')
        else:
            flow_direction.append('anterograde')


    # add data to the dataframes
    myosin_trackdata_df['flow_angle'] = flow_angle
    myosin_trackdata_df['flow'] = flow_direction
    myosin_trackdata_df['flow_switch'] = flow_switch
    myosin_trackdata_df['flow_switch_position'] = flow_switch_position
    myosin_trackdata_df['flow_switch_travel'] = flow_switch_travel

    switched_myosin_trackdata_df = myosin_trackdata_df[myosin_trackdata_df['flow_switch'] == 'switches']

    if len(switched_myosin_trackdata_df) > 0:
        switching_myosin_df = break_up_switching_tracks(switched_myosin_trackdata_df, frame_interval=frame_interval, median_angle = median_angle)

        myosin_trackdata_df = myosin_trackdata_df[myosin_trackdata_df['flow_switch'] == 'straight']
        myosin_trackdata_df = pd.concat([myosin_trackdata_df, switching_myosin_df])

    # make anterograde and retrograde dataframes
    anterograde_myosin_df = myosin_trackdata_df[myosin_trackdata_df['flow'] == 'anterograde']
    retrograde_myosin_df = myosin_trackdata_df[myosin_trackdata_df['flow'] == 'retrograde']

    # make a list of particles in each dataframe
    anterograde_myosin_particles = list(anterograde_myosin_df['particle'])
    retrograde_myosin_particles = list(retrograde_myosin_df['particle'])

    return myosin_trackdata_df, anterograde_myosin_particles, anterograde_myosin_df, retrograde_myosin_particles, retrograde_myosin_df

def plot_centered_trajectories(myosin_trackdata_df, filename, anterograde_color = '#d58440', retrograde_color = '#42749d', smooth_trajectories = False, save_image = True):

    retro_velocity, antero_velocity = [],[]
    fig, ax = plt.subplots(ncols=2,figsize=(12, 4))
    for index, row in myosin_trackdata_df.iterrows():
        if smooth_trajectories:
            x = row['x_smoothed']
            y = -1*row['y_smoothed']
        else:
            x = row['x']
            y = -1*row['y']
        
        if row['flow'] == 'anterograde':
            ax[0].plot(x,y,color=anterograde_color)
            if smooth_trajectories:
                antero_velocity.append(row['actual_velocity_smoothed'])
            else:
                antero_velocity.append(row['actual_velocity'])
        else:
            ax[0].plot(x,y,color=retrograde_color)
            if smooth_trajectories:
                retro_velocity.append(row['actual_velocity_smoothed'])
            else:
                retro_velocity.append(row['actual_velocity'])

    ax[0].axis('equal')        
    ymin, ymax = ax[0].get_ylim()
    xmin, xmax = ax[0].get_xlim()
    ax[0].plot([0,0],[ymin,ymax],'-.k')
    ax[0].plot([xmin,xmax],[0,0], '-.k')


    mean_antero_velocity = np.mean(antero_velocity)
    mean_retro_velocity = np.mean(retro_velocity)
    orange_patch = mpatches.Patch(color=anterograde_color, label='Anterograde Flow (' + str(np.round(mean_antero_velocity,2)) + ' nm/s)')
    blue_patch = mpatches.Patch(color=retrograde_color, label='Retrograde Flow (' + str(np.round(mean_retro_velocity,2)) + ' nm/s)')
    ax[0].legend(handles=[blue_patch, orange_patch])
    ax[0].set_xlabel('nm')
    ax[0].set_ylabel('nm')



    my_pal = {"retrograde": retrograde_color, "anterograde": anterograde_color}
    sns.boxplot(data = myosin_trackdata_df, x = 'flow', y='effective_velocity', palette=my_pal, ax=ax[1])
    sns.swarmplot(data = myosin_trackdata_df, x = 'flow', y='effective_velocity', color='k', alpha= 0.5, size = 3, ax=ax[1])
    orange_patch = mpatches.Patch(color=anterograde_color, label='Anterograde Flow')
    blue_patch = mpatches.Patch(color=retrograde_color, label='Retrograde Flow')
    ax[1].legend(handles=[blue_patch, orange_patch])
    ax[1].set_ylabel('velocity (nm/s)')
    fig.tight_layout()

    fig.show()

    if save_image:
        fig.savefig(filename[:-4] + '_myosin_puncta_tracked.png', dpi=300)

    return

def make_tracked_movie_stack(filename, imstack, myosin_tracks_filtered, anterograde_myosin_particles, retrograde_myosin_particles, max_inten = None, min_inten = None):
    
    movie_folder = filename[:filename.rfind('/')] + '/movie'
    # make a directory to store all the displacement files
    if os.path.isdir(movie_folder) == False:
        os.mkdir(movie_folder)
    N_images = imstack.shape[0]
    fig, ax = plt.subplots()
    fig.show()
    for frame in np.arange(0,N_images):
        ax.clear()
        ax.imshow(imstack[frame], vmin=min_inten, vmax=max_inten, cmap='Greys')
        frame_df = myosin_tracks_filtered[myosin_tracks_filtered['frame'] == frame]
        for index, row in frame_df.iterrows():
            if row['particle'] in anterograde_myosin_particles:
                ax.scatter(row['x'], row['y'], s=100, facecolors='none', edgecolors=anterograde_color)
                ax.text(row['x']+4, row['y'], str(int(row['particle'])), color=anterograde_color)
            elif row['particle'] in retrograde_myosin_particles:
                ax.scatter(row['x'], row['y'], s=100, facecolors='none', edgecolors=retrograde_color)
                ax.text(row['x']+4, row['y'], str(int(row['particle'])), color=retrograde_color)
        ax.axis('off')
        fig.savefig(movie_folder + '/Myosin_flow_frame_%03d.png' % frame, dpi = 150)


    file_list = sorted(glob.glob(movie_folder + '/Myosin_flow_frame*.png'))
    Nframes = len(file_list)
    first_frame = io.imread(movie_folder + '/Myosin_flow_frame_000.png')
    first_frame = first_frame[:,:,0]
    rows, cols = [],[]
    for i in np.arange(0,first_frame.shape[0]):
        rows.append(len(np.unique(first_frame[i,:])))
    for j in np.arange(0,first_frame.shape[1]):
        cols.append(len(np.unique(first_frame[:,j])))

    rows_withdata = np.where(np.array(rows) > 1)
    cols_withdata = np.where(np.array(cols) > 1)

    row_begin = rows_withdata[0][0]
    row_end = rows_withdata[0][-1] + 1
    col_begin = cols_withdata[0][0]
    col_end = cols_withdata[0][-1] + 1

    plot_stack = np.zeros((Nframes, row_end - row_begin, col_end - col_begin,3), dtype = 'uint8')
    for i,im_name in enumerate(file_list):
        im = io.imread(im_name)
        plot_stack[i] = im[row_begin:row_end,col_begin:col_end,0:3]

    io.imsave(filename[:-4] + '_movie.tif', plot_stack.astype('uint8'))  
    #remove the individual images
    shutil.rmtree(movie_folder)

    return

def break_up_switching_tracks(switched_myosin_trackdata_df, frame_interval = 1, median_angle = 0, retrograde_color='#42749d', anterograde_color='#d58440'):

    particle, pathlength, displacement, actual_velocity, effective_velocity, costheta, mean_costheta, duration = [],[],[],[],[],[],[],[]
    x, y, frames, x_smoothed, y_smoothed, frames_smoothed, pathlength_smoothed, displacement_smoothed = [],[],[],[],[],[],[],[]
    actual_velocity_smoothed, effective_velocity_smoothed, duration_smoothed = [],[],[]
    flow_angle, flow, flow_switch, flow_switch_position, flow_switch_travel = [],[],[],[],[]

    for index, row in switched_myosin_trackdata_df.iterrows():
        x_smoothed_pt1 = row['x_smoothed'][:row['flow_switch_position']]
        y_smoothed_pt1 = row['y_smoothed'][:row['flow_switch_position']]
        frames_smoothed_pt1 = row['frames_smoothed'][:row['flow_switch_position']]
        
        x_smoothed_pt2 = row['x_smoothed'][row['flow_switch_position']:]
        y_smoothed_pt2 = row['y_smoothed'][row['flow_switch_position']:]
        x_smoothed_pt2 = x_smoothed_pt2 - x_smoothed_pt2[0]
        y_smoothed_pt2 = y_smoothed_pt2 - y_smoothed_pt2[0]
        frames_smoothed_pt2 = row['frames_smoothed'][row['flow_switch_position']:]
        
        path_length_pt1,path_duration_pt1, particle_velocity_pt1, actual_displacement_pt1, effective_velocity_pt1 = calculate_length_and_velocity(x_smoothed_pt1, y_smoothed_pt1, frames_smoothed_pt1, frame_duration=frame_interval)
        path_length_pt2,path_duration_pt2, particle_velocity_pt2, actual_displacement_pt2, effective_velocity_pt2 = calculate_length_and_velocity(x_smoothed_pt2, y_smoothed_pt2, frames_smoothed_pt2, frame_duration=frame_interval)
        
        angle_pt1 = np.arctan2(y_smoothed_pt1[-1], x_smoothed_pt1[-1])
        angle_pt2 = np.arctan2(y_smoothed_pt2[-1], x_smoothed_pt2[-1])
        
        x_median = np.cos(median_angle)
        y_median = np.sin(median_angle)
        
        if (x_median*x_smoothed_pt1[-1] - y_median*y_smoothed_pt1[-1]) >= 0:
            flow_direction_pt1 = 'retrograde'
        else:
            flow_direction_pt1 = 'anterograde'
        if (x_median*x_smoothed_pt2[-1] - y_median*y_smoothed_pt2[-1]) >= 0:
            flow_direction_pt2 = 'retrograde'
        else:
            flow_direction_pt2 = 'anterograde'
        
        # path 1
        particle.append(row['particle'])
        pathlength.append(row['pathlength'])
        displacement.append(row['displacement'])
        actual_velocity.append(row['actual_velocity'])
        effective_velocity.append(row['effective_velocity'])
        costheta.append(row['costheta'])
        mean_costheta.append(row['mean_costheta'])
        duration.append(row['duration'])
        x.append(row['x'])
        y.append(row['y'])
        frames.append(row['frames'])
        x_smoothed.append(x_smoothed_pt1)
        y_smoothed.append(y_smoothed_pt1)
        frames_smoothed.append(frames_smoothed_pt1)
        pathlength_smoothed.append(path_length_pt1)
        displacement_smoothed.append(actual_displacement_pt1)
        actual_velocity_smoothed.append(particle_velocity_pt1)
        effective_velocity_smoothed.append(effective_velocity_pt1)
        duration_smoothed.append(path_duration_pt1)
        flow_angle.append(angle_pt1)
        flow.append(flow_direction_pt1)
        flow_switch.append(row['flow_switch'])
        flow_switch_position.append(row['flow_switch_position'])
        flow_switch_travel.append(row['flow_switch_travel'])
        
        # path 2
        particle.append(row['particle'])
        pathlength.append(row['pathlength'])
        displacement.append(row['displacement'])
        actual_velocity.append(row['actual_velocity'])
        effective_velocity.append(row['effective_velocity'])
        costheta.append(row['costheta'])
        mean_costheta.append(row['mean_costheta'])
        duration.append(row['duration'])
        x.append(row['x'])
        y.append(row['y'])
        frames.append(row['frames'])
        x_smoothed.append(x_smoothed_pt2)
        y_smoothed.append(y_smoothed_pt2)
        frames_smoothed.append(frames_smoothed_pt2)
        pathlength_smoothed.append(path_length_pt2)
        displacement_smoothed.append(actual_displacement_pt2)
        actual_velocity_smoothed.append(particle_velocity_pt2)
        effective_velocity_smoothed.append(effective_velocity_pt2)
        duration_smoothed.append(path_duration_pt2)
        flow_angle.append(angle_pt2)
        flow.append(flow_direction_pt2)
        flow_switch.append(row['flow_switch'])
        flow_switch_position.append(row['flow_switch_position'])
        flow_switch_travel.append(row['flow_switch_travel'])
        
    switching_myosin_dict = {
        'particle' : particle,
        'pathlength' : pathlength,
        'displacement' : displacement,
        'actual_velocity' : actual_velocity,
        'effective_velocity' : effective_velocity,
        'costheta' : costheta,
        'mean_costheta' : mean_costheta,
        'duration' : duration,
        'x' : x,
        'y' : y,
        'frames' : frames,
        'x_smoothed' : x_smoothed,
        'y_smoothed' : y_smoothed,
        'frames_smoothed' : frames_smoothed,
        'pathlength_smoothed' : pathlength_smoothed,
        'displacement_smoothed' : displacement_smoothed,
        'actual_velocity_smoothed' : actual_velocity_smoothed,
        'effective_velocity_smoothed' : effective_velocity_smoothed,
        'duration_smoothed' : duration_smoothed,
        'flow_angle' : flow_angle,
        'flow' : flow,
        'flow_switch' : flow_switch,
        'flow_switch_position' : flow_switch_position,
        'flow_switch_travel' : flow_switch_travel
    }

    switching_myosin_df = pd.DataFrame(switching_myosin_dict)

    return  switching_myosin_df

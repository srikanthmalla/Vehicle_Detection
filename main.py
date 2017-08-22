from src.feature_extract import *
from src.load_data import *
from src.classifier import *
from src.slide_window import *
from src.params import *
from src.helper_funcs import *
import scipy.misc
#Sliding Window Test on the Images
t=time.time() # Start time
for image_p in glob.glob('test_images/test*.jpg'):
    image = cv2.imread(image_p)
    draw_image = np.copy(image)
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 640], 
                    xy_window=(128, 128), xy_overlap=(0.85, 0.85))
    hot_windows = []
    hot_windows += (search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat))                       
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
    show_img(window_img)
    plt.show()
    scipy.misc.imsave('output_'+image_p, cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB))    

print(round(time.time()-t, 2), 'Seconds to process test images')

##   Advanced Sliding Window Test
image = cv2.imread('test_images/test5.jpg')
windows = slide_window(image, x_start_stop=[900, None], y_start_stop=[420, 650], 
                    xy_window=(128, 128), xy_overlap=(0.75, 0.75))
windows += slide_window(image, x_start_stop=[0, 350], y_start_stop=[420, 650], 
                    xy_window=(128, 128), xy_overlap=(0.75, 0.75))
window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6) 
windows = slide_window(image, x_start_stop=[400, 880], y_start_stop=[400, 470], 
                    xy_window=(48, 48), xy_overlap=(0.75, 0.75))
window_img = draw_boxes(window_img, windows, color=(255, 0, 0), thick=6)                    
show_img(window_img)
plt.show()
scipy.misc.imsave('output_test_images/window.jpg', cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB))    


#Video Pipeline
# from moviepy.editor import VideoFileClip
# n_count = 0
# def process_image(image):
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return cv2.cvtColor(frame_proc(image, lane=False, video=True, vis=False), cv2.COLOR_BGR2RGB)

# output_v = 'project_video_proc.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# clip = clip1.fl_image(process_image)
# clip.write_videofile(output_v, audio=False)
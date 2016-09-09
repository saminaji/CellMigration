
import cv2
from math import cos, sin
from math import cos, sin, sqrt
import numpy as np
import sys

if __name__ == "__main__":

    A = [ [1, 1], [0, 1] ]


    img_height = 500
    img_width = 500
    kalman = cv2.KalmanFilter(2, 1, 0)
    process_noise = cv.CreateMat(2, 1, cv.CV_32FC1)
    measurement = cv.CreateMat(1, 1, cv.CV_32FC1)

    code = -1L


    cv2.namedWindow("Kalman")


 	    while True:
    	        state = 0.1 * np.random.randn(2, 1)

   	 	        kalman.transitionMatrix = np.array([[1., 1.], [0., 1.]])
    	 	        kalman.measurementMatrix = 1. * np.ones((1, 2))
        	        kalman.processNoiseCov = 1e-5 * np.eye(2)
     	 	        kalman.measurementNoiseCov = 1e-1 * np.ones((1, 1))
 	        kalman.errorCovPost = 1. * np.ones((2, 2))
    	        kalman.statePost = 0.1 * np.random.randn(2, 1)

        while True:
                  def calc_point(angle):
                    return (np.around(img_height/2 - img_width/3*sin(angle), 1).astye(int))

            	 	            state_angle = state[0, 0]
            	 	            state_pt = calc_point(state_angle)

            	 	            prediction = kalman.predict()
           	 	            predict_angle = prediction[0, 0]
           	 	            predict_pt = calc_point(predict_angle)



            # generate measurement	 	            # generate measurement
            cv.MatMulAdd(kalman.measurement_matrix, state, measurement, measurem
ent)	 	            measurement = np.dot(kalman.measurementMatrix, state) + measurement

            measurement_angle = measurement[0, 0]	 	            measurement_angle = measurement[0, 0]
            measurement_pt = calc_point(measurement_angle)	 	            measurement_pt = calc_point(measurement_angle)

            # plot points	 	            # plot points
      	 	            def draw_cross(center, color, d):
              	 	                cv2.line(img,
                                             (center[0] - d, center[1] - d), (center[0] + d, center[
 0)	 	1] + d),
                cv.Line(img, (center[0] + d, center[1] - d),	 	                         color, 1, cv2.LINE_AA, 0)
                             (center[0] - d, center[1] + d), color, 1, cv.CV_AA,	 	                cv2.line(img,
 0)	 	                         (center[0] + d, center[1] - d), (center[0] - d, center[
                                                                                	 	1] + d),
            cv.Zero(img)	 	                         color, 1, cv2.LINE_AA, 0)
            draw_cross(state_pt, cv.CV_RGB(255, 255, 255), 3)
            draw_cross(measurement_pt, cv.CV_RGB(255, 0,0), 3)	 	            img = np.zeros((img_height, img_width, 3), np.uint8)
            draw_cross(predict_pt, cv.CV_RGB(0, 255, 0), 3)	 	            draw_cross(np.int32(state_pt), (255, 255, 255), 3)
            cv.Line(img, state_pt, measurement_pt, cv.CV_RGB(255, 0,0), 3, cv. C	 	            draw_cross(np.int32(measurement_pt), (0, 0, 255), 3)
V_AA, 0)	 	            draw_cross(np.int32(predict_pt), (0, 255, 0), 3)
            cv.Line(img, state_pt, predict_pt, cv.CV_RGB(255, 255, 0), 3, cv. CV
_AA, 0)	 	            cv2.line(img, state_pt, measurement_pt, (0, 0, 255), 3, cv2.LINE_AA,
 	 0)
            cv.KalmanCorrect(kalman, measurement)	 	            cv2.line(img, state_pt, predict_pt, (0, 255, 255), 3, cv2.LINE_AA, 0
 	)
            cv.RandArr(rng, process_noise, cv.CV_RAND_NORMAL, cv.RealScalar(0),
                       cv.RealScalar(sqrt(kalman.process_noise_cov[0, 0])))	 	            kalman.correct(measurement)
            cv.MatMulAdd(kalman.transition_matrix, state, process_noise, state)
 	            process_noise = kalman.processNoiseCov * np.random.randn(2, 1)
 	            state = np.dot(kalman.transitionMatrix, state) + process_noise

            	            cv2.imshow("Kalman", img)

         	 	            code = cv2.waitKey(100) % 0x100
            	 	            if code != -1:
                	 	                break

       	 	        if code in [27, ord('q'), ord('Q')]:
          	 	            break

	 	    cv2.destroyWindow("Kalman")
import cv2
import numpy as np

# Weights of the Gaussian Blur
a = np.array([1/16, 1/4, 3/8, 1/4, 1/16])
b = np.array([[1/16], [1/4], [3/8], [1/4], [1/16]])
Weights = a*b

Iteration = [-2, -1, 0, 1, 2]


def Gaussian_Pyramid(Input):

   w, h = Input.shape[:2]

   Gaussian_Image = np.zeros((int(w/2),int(h/2)))

   for index in range(int(w/2)):
      for itr in range(int(h/2)):
        for var in range(5):
           for cor in range(5):
             Gaussian_Image[index][itr] += Weights[var][cor]*Input[2*index - var][2*itr - cor]
             
   return Gaussian_Image
   
   
def Laplacian_Pyramid(Input):

   w, h = Input.shape[:2]

   Laplacian_Image = np.zeros((int(w*2),int(h*2)))

   for index in range(int(w*2)):
      for itr in range(int(h*2)):
             Laplacian_Image[index][itr] += Input[int(index/2)][int(itr/2)]
             
   return Laplacian_Image
   
   
def Pyramid_Image_For_Level(Image_List):
    Display_Image = np.ones((w, int(1.5*h)))
    width_Num = 0
    height_Num = 0
    for index in range(len(Image_List)):
    
        Image_Disp = Image_List[index]
        tmpW, tmpH = Image_Disp.shape[:2]
        
        if(index != 0):
           height_Num = h
        
        if(index == 2):
           width_Num = int(w/2)
           
        if(index == 3):
           width_Num = int(w/2 + w/4)
        
        if(index == 4):
           width_Num = int(w/2 + w/4 + w/8)
           
        if(index == 5):
           width_Num = int(w/2 + w/4 + w/8 + w/16)
            
        for i in range(tmpW):
           for j in range(tmpH):
              Display_Image[i+width_Num][j+height_Num] = Image_Disp[i][j]/256
    
    return Display_Image

Original_Name = "image.jpg"
Original = cv2.imread(Original_Name ,cv2.IMREAD_GRAYSCALE)
w, h = Original.shape[:2]

current = Original

Gaussian_Images = [Original]
Laplacian_Images = []

for cycle in range(5):

   # Build Gaussian Pyramid
   print("Started Gaussian Pyramid")
   result = Gaussian_Pyramid(current)

   filename = Original_Name[:-4]+"_GP_Level_"+ str(cycle+1) +".jpg"
   cv2.imwrite(filename, result)
   Gaussian_Images.append(result)
   
   #Build Laplacian Pyramid
   print("Started Laplacian Pyramid")
   Laplacian = Laplacian_Pyramid(result)
   
   # Get the difference
   print("Get the Difference Image")
   Cw, Ch = current.shape[:2]
   Lw, Lh = Laplacian.shape[:2]
   
   # Image resize in cases where width or height aren't factor of 2^5
   if(Cw != Lw or Ch != Lh):
      Laplacian = cv2.resize(Laplacian, (Ch, Cw))
   
   Diff = current - Laplacian

   diffname = Original_Name[:-4]+"_Diff_Level_"+ str(cycle+1) +".jpg"
   cv2.imwrite(diffname, Diff) 
   Laplacian_Images.append(Diff)

   current = result

G_Display_Image = Pyramid_Image_For_Level(Gaussian_Images)
L_Display_Image = Pyramid_Image_For_Level(Laplacian_Images)
      
cv2.imshow( "Gaussian Pyramid", G_Display_Image )
GP_Name = Original_Name[:-4]+"_Gaussian_Pyramid.jpg"
cv2.imwrite(GP_Name, G_Display_Image)

cv2.imshow( "Laplacian Pyramid", L_Display_Image )
LP_Name = Original_Name[:-4]+"_Laplacian_Pyramid.jpg"
cv2.imwrite(LP_Name, L_Display_Image)


cv2.waitKey(0)

cv2.destroyAllWindows()


print("..........END..........") 

  





# OVO JE py FAJL ZA CITANJE KODA
# ISTI KOD I TO SA REZULTATIMA SE NALAZI U Jupyter Notebook spripti
# ALI GA NIJE MOGUCE PRIKAZATI ZBOG NJEGOVE VELICINE 


# ucitavanje biblioteka potrebnih za rad
from __future__ import print_function
from pylab import *

import skimage 
from skimage  import color, filters, img_as_float, morphology, feature

# funkcije za racunanje Hafove transformacije
from skimage.transform import hough_line, hough_line_peaks

# za poredjenje implementiranog i ugradjenog Kanijevog algoritma
from skimage.metrics import structural_similarity as ssim 

import scipy 
from scipy import ndimage

import numpy as np

import time
from time import time

# biblioteka upotrebljena za iscrtavanje linija na slikama
import cv2 

# biblioteka upotrebljena za dohvatanje potrebnih putanja foldera
import os

folder_path = 'sekvence/' # root folder za slike i video



######################################################## 
def segment_white_lanes(img_in):
    
    """
    Opis: 
        Funkcija uzima ulaznu sliku u RGB kolor sistemu, prebacuje je u 
        HSV kolor sistem i na osnovu eksperimentalno odredjenih pragova 
        za komponente S i V vrsi segmentaciju belih delova slike. 
        Primaran cilj je sto bolja segmentacija bele kolovozne trake na putu.
        
    Parametri:
        img_in - slika u RGB kolor sistemu
    
        
    Funkcija vraca binarnu masku ulazne slike, gde su beli regioni beli delovi ulazne slike i
    pored toga sve ostale slike i vrednosti potrebne za iscrtavanje rezultata.
    
    """
    # izdvajanje dimenzija slike
    [M, N, D] = shape(img_in)
    
    # koriste se sve komponente HSV kolor sistema osim H, 
    # posto belu boju ne mozemo fino preko H komponente da izdvojimo
    s_mask = zeros((M,N))
    v_mask = zeros((M,N))
    img_out = zeros((M,N))

    # prebacivanje RGB ulazne slike u HSV kolor sistem
    img_hsv = color.rgb2hsv(img_in)
    
    # pragovi za S i V komponentu
    # bela boja ima vrlo malu saturaciju, pa je s_threshold najveca moguca
    # vrednost S ulazne slike, sa druge strane V komponenta bele boje
    # je velika, pa je v_threshold donja granica komponente V za ulaznu sliku
    s_threshold = 0.2
    v_threshold = 0.9
    
    # formiranje S maske
    s_mask = img_hsv[:,:,1] < s_threshold
    # formiranje V maske
    v_mask = img_hsv[:,:,2] > v_threshold
    
    # formiranje krajnje (izlazne) maske
    img_out = s_mask * v_mask
    
    return [img_hsv, s_mask, s_threshold, v_mask, v_threshold, img_out]

def segment_yellow_lanes(img_in):
    
    """
    Opis: 
        Funkcija uzima ulaznu sliku u RGB kolor sistemu, prebacuje je u 
        HSV kolor sistem i na osnovu eksperimentalno odredjenih pragova 
        za komponente H, S i V vrsi segmentaciju belih delova slike. 
        Primaran cilj je sto bolja segmentacija zute kolovozne trake na putu.
        
    Parametri:
        img_in - slika u RGB kolor sistemu
    
        
    Funkcija vraca binarnu masku ulazne slike, gde su beli regioni zuti delovi ulazne slike i
    pored toga sve ostale slike i vrednosti potrebne za iscrtavanje rezultata.
    
    """
    # izdvajanje dimenzija slike
    [M, N, D] = shape(img_in)
    
    # koriste se sve komponente HSV kolor sistema
    # posto zutu boju mozemo fino opsegom H komponente da izdvojimo
    # a sa komponentama S i V pratimo vrednosti njenog sjaja i jacine
    h_mask = zeros((M,N))
    s_mask = zeros((M,N))
    v_mask = zeros((M,N))
    img_out = zeros((M,N))

    # prebacivanje RGB ulazne slike u HSV kolor sistem
    img_hsv = color.rgb2hsv(img_in)
    
    # eksperimentalno odredjeni pragovi za svaku od komponenti
    # postoje 2 praga za H komponentu - H komponenta za zutu boju
    # je otprilike u opsegu od 20 do 50 stepeni na "HSV valjku" 
    # ali su ti pragovi pomereni malo prema zelenoj i malo prema crvenoj boji
    h_threshold_1 = 20/360
    h_threshold_2 = 65/360
    
    # takodje, postavljena su i dva praga za S komponentu
    # jarko zuta boja ima veliku S komponentu, pa je uzet deo visih vrednosti S komponente
    s_threshold_1 = 0.5
    s_threshold_2 = 0.85
    # V komponenta je takodje velika kod zute boje, ali da bi se dobro izdvojila
    # zuta traka, ovaj prag nije postavljen previsoko, posto sve maske koje se forimiraju od ovih
    # pragova treba se "enduju", pa da se ne bi izgubila nekad informacija
    v_threshold = 0.85
    
    # formiranje H maske
    h_mask = (img_hsv[:,:,0] > h_threshold_1) * (img_hsv[:,:,0] < h_threshold_2)
    # formiranje S maske
    s_mask = (img_hsv[:,:,1] > s_threshold_1) * (img_hsv[:,:,1] < s_threshold_2)
    # formiranje V maske
    v_mask = img_hsv[:,:,2] > v_threshold
    
    # formiranje krajnje (izlazne) maske
    img_out = h_mask * s_mask * v_mask
    
    return  [img_hsv, h_mask, h_threshold_1, h_threshold_2, s_mask, s_threshold_1, s_threshold_2 , v_mask, v_threshold, img_out]

def segment_lanes(img_in, interal_plot = False):
    
    """
    Opis: 
        Funkcija uzima ulaznu sliku i iz slike izdvaja 
        regione koji predstavljaju zutu (levu) i belu (desnu) kolovoznu
        traku.
        
    Parametri:
        img_in - ulazna slika
    
        
    Funkcija vraca binarnu masku ulazne slike, gde su beli regioni beli i zuti delovi ulazne slike,
    koji uglavnom predstavljaju kolovozne trake na putu.
    
    """
    # izdvajanje dimenzija slike
    [M, N, D] = shape(img_in)
        
    # inicijalizacija izlazne binarne maske
    img_out = zeros((M, N))
    
    # poziv funkcija za izdvajanje bele i zute trake
    [hsv_w, s_w, s_t_w, v_w, v_t_w, img_out_white] = segment_white_lanes(img_in)
    [hsv_y, h_y, h_t_1_y, h_t_2_y, s_y, s_t_1_y, s_t_2_y , v_y, v_t_y, img_out_yellow] = segment_yellow_lanes(img_in)
    
    # izlazna binarna maska je zbir dve maske, svake za po jednu kolovoznu traku
    img_out = img_out_white + img_out_yellow
    
    # pored ovoga, pokusano je poboljsanje ovih izdvojenih regiona upotrebom median filtra i binarne dilatacije
    # i to na sledeci nacin:
    #     img_out = ndimage.median_filter(img_out, size=(5,5), mode='mirror')
    #     img_out = morphology.binary_dilation(img_out, morphology.disk(3))  
    # nazalost, ovi filtri su u pojedinim momentima prejaki, pa dolazi do nestajanja nekih slabih regiona
    # koji su bitni za zadatak, te je to razlog zbog kojeg nisu upotrebljene ove metode
    
    # iscrtavanje rezultata segmentacije kolovoznih traka
    if interal_plot :
        fig, ax = plt.subplots(nrows = 6, ncols = 2, figsize = (16,24), dpi = 120)
        
        ax[0,0].imshow(img_in, cmap = 'jet')
        ax[0,0].set_title('Ulazna slika', fontsize = 16)
        ax[0,0].set_axis_off()

        ax[0,1].imshow(hsv_w[:,:,0], cmap = 'hsv')
        ax[0,1].set_title('H komponenta', fontsize = 16)
        ax[0,1].set_axis_off()
        
        ax[1,0].imshow(hsv_w[:,:,1], cmap = 'jet')
        ax[1,0].set_title('S komponenta', fontsize = 16)
        ax[1,0].set_axis_off()
        
        ax[1,1].imshow(hsv_w[:,:,2], cmap = 'jet')
        ax[1,1].set_title('V komponenta', fontsize = 16)
        ax[1,1].set_axis_off()
        
        ax[2,0].imshow(s_w, cmap = 'gray')
        ax[2,0].set_title('S maska za bele regione \n s_threshold = ' + str(s_t_w), fontsize = 16)
        ax[2,0].set_axis_off()
        
        ax[2,1].imshow(v_w, cmap = 'gray')
        ax[2,1].set_title('V maska za bele regione \n v_threshold = ' + str(v_t_w), fontsize = 16)
        ax[2,1].set_axis_off()
        
        ax[3,0].imshow(img_out_white, cmap = 'gray')
        ax[3,0].set_title('Segmentisani beli regioni', fontsize = 16)
        ax[3,0].set_axis_off()
        
        ax[3,1].imshow(h_y, cmap = 'gray')
        ax[3,1].set_title('H maska za zute regione \n h_threshold_low = ' + "{:.3f}".format(h_t_1_y) + ' h_threshold_high = ' + "{:.3f}".format(h_t_2_y), fontsize = 16)
        ax[3,1].set_axis_off()
        
        ax[4,0].imshow(s_y, cmap = 'gray')
        ax[4,0].set_title('S maska za zute regione \n s_threshold_low = ' + str(s_t_1_y) + ' s_threshold_high = ' + str(s_t_2_y), fontsize = 16)
        ax[4,0].set_axis_off()
        
        ax[4,1].imshow(v_y, cmap = 'gray')
        ax[4,1].set_title('V maska za zute regione \n v_threshold = ' + str(v_t_y), fontsize = 16)
        ax[4,1].set_axis_off()
        
        ax[5,0].imshow(img_out_yellow, cmap = 'gray')
        ax[5,0].set_title('Segmentisani zuti regioni', fontsize = 16)
        ax[5,0].set_axis_off()
        
        ax[5,1].imshow(img_out, cmap = 'gray')
        ax[5,1].set_title('Krajnja maska regiona od znacaja', fontsize = 16)
        ax[5,1].set_axis_off()
        
        plt.tight_layout()
        plt.show()
        
    return img_out

###########################################################

print('Testiranje segmentacije kolovoznih traka na ulaznim test slikama :\n')
for i in range(1,7,1):
    img_name = 'test'+str(i)+'.jpg'
    img_in = img_as_float(imread(folder_path + img_name))
    
    print('----- Slika ' + img_name + ' -----')
    
    img_seg = segment_lanes(img_in, True)   

###########################################################

def sobel_gradient(image):
    
    """
    Opis: 
        Funkcija uzima ulaznu sliku i racuna hoziontalni i vertikalni
        gradijent slike, kao i magnitudu i ugao gradijenta slike pomocu Sobelovog
        gradijentnog operatora.
        
    Parametri:
        image - ulazna slika u grayscale formatu
    
    Funkcija vraca 4 vrednosti - horizontalni gradijent slike, vertikalni gradijent slike,
    magnitudu gradijenta i ugao gradijenta slike.
    
    """
    # maska Sobelovog operatora za horizontalni gradijent slike
    Hx = [[-1, -2, -1],[0, 0, 0],[1, 2, 1]]
    
    # transponovana maska Hx za vertikalni gradijent slike
    Hy = np.transpose(Hx)
    
    # izracunavanje vertikalnog i horizontalnog gradijenta slike
    Gx = ndimage.convolve(image, Hx, mode='mirror')
    Gy = ndimage.convolve(image, Hy, mode='mirror')
    
    # izracunavanje magnitude gradijenta slike
    mag = np.sqrt(np.square(Gx) + np.square(Gy))
    
    # izracunavanje ugla gradijenta slike sa dodatnom pomocnom promenljivom
    # kako ne bi izlazilo upozorenje za deljenje sa nulom
    Gx_fix = np.copy(Gx)
    Gx_fix[Gx_fix==0] = 1
    angle = np.arctan(np.divide(Gy,Gx_fix))
    angle[(Gx==0)&(Gy>0)] = np.pi/2
    angle[(Gx==0)&(Gy<0)] = -np.pi/2
    angle[(Gx==0) & (Gy==0)] = 0
    
    # pretvaranje ugla slike u stepene, radi lakse kvantizacije
    angle = 180*angle/pi
    
    return [Gx, Gy, mag, angle]

def angle_quantization(angle_grad):
    
    """
    Opis: 
        Funkcija uzima ulazne uglove gradijenta slike dobijene primenom Sobelovog
        operatora na nju i vrsi kvantizaciju nad njima na 4 pravca : 0 stepeni, 45 stepeni
        90 stepeni, i -45 stepeni.
        
    Parametri:
        angle_grad - uglovi gradijenta slike dobijeni primenom Sobelovog operatora na sliku
    
    Funkcija vraca kvantizovana uglove, tj. sliku na kojoj se na svakom pikselu nalazi 
    vrednost kvantizovanog ugla gradijenta.
    
    """
    # kvantizacija uglova 
    angle_quant = angle_grad.copy()
    angle_quant[(angle_grad > -22.5) & (angle_grad < 22.5)] = 0
    angle_quant[(angle_grad > -67.5) & (angle_grad <= -22.5)] = 45
    angle_quant[(angle_grad >= 22.5) & (angle_grad < 67.5)] = -45
    angle_quant[(angle_grad >= 67.5) | (angle_grad <= -67.5)] = 90
    
    return angle_quant

def local_non_max_suppression(mag, angle):
    
    """
    Opis: 
        Funkcija vrsi potiskivanje lokalnih ne-maksimuma na slici koja predstavlja
        magnitudu gradijenta originalne slike. Potiskivanje se vrsi po uglovima/pravcima
        gradijenta koji su definisane u ulaznoj matrici angle.
        
    Parametri:
        mag - ulazna matrica/slika magnituda gradijenta za potiskivanje lokalnih ne-maksimuma
        angle - ulazna matrica/slika sa uglovima kvantizovanim na 4 pravca po kojoj se 
                vrsi potiskivanje lokalnih ne-maksimuma
    
    Funkcija vraca matricu/sliku svih magnituda koje su u svom lokalnom susedstvu, 
    definisanim uglom/pravcem prostiranja gradijenta, najizrazenije, odnosno maksimalne.
    
    """
    # dohvatanje dimenzija matrice/slike
    [M,N] = shape(mag)
    
    # prosirivanje matrice magnituda sa svih strana za po 1 piksel
    # ovo se radi, jer se za svaki piksel provera maltene 8-susedstvo,
    # pa da bi se ta provera izvrsila i nad pikselima na ivicama slike, 
    # slika se mora prosiriti za po 1 piksel sa svake strane
    mag_pad = zeros((M+2,N+2))
    mag_pad[1:M+1, 1:N+1] = mag.copy()
    
    # takodje se prosiruje i matrica/slika uglova
    # razlika je samo sto se ova matrica ne prosiruje nulama kao matrica
    # magnituda, jer nula kao ugao predstavlja vertikalan pravac ivice, pa je 
    # zbog toga ova matrica prosirena jedinicama kao neutralnim uglom
    angle_pad = ones((M+2, N+2))
    angle_pad[1:M+1, 1:N+1] = angle.copy()
    
    # incijalizacija matrica lokalnih maksimuma
    local_mag_max = zeros_like(mag_pad)
    
    # nalazenje indeksa u matrici uglova koja imaju odgovarajuce 
    # kvantizovane uglove gradijenta
    # ugradjena funkcija iz biblioteke numpy - argwhere, trazi
    # elementa matrice koji imaju prosledjenu vrednost, i vraca 
    # njihove indekse u matrici
    angle_0_ind = np.argwhere(angle_pad == 0)
    angle_45_ind = np.argwhere(angle_pad == 45)
    angle_90_ind = np.argwhere(angle_pad == 90)
    angle_minus_45_ind = np.argwhere(angle_pad == -45)
    
    # dakle, ideja je da se ne prolazi kroz 2 ugnjezdene for-petlje za svaki piksel,
    # vec da se prolazi kroz 4 manje for petlje i da se vrsi provera lokalnog
    # susedstva
    
    # for petlja za uglove od 0 stepeni
    for i in range(len(angle_0_ind)):
        x = angle_0_ind[i][0]
        y = angle_0_ind[i][1]
        
        # provera lokalnog susedstva za vertikalne ivice
        if (mag_pad[x,y] > mag_pad[x-1,y]) & (mag_pad[x,y] > mag_pad[x+1,y]):
            # cuvanje vrednosti ukoliko je ona maksimalna u svom lok. susedstvu
            local_mag_max[x,y] = mag_pad[x,y]
    
    # for petlja za uglove od 45 stepeni
    for i in range(len(angle_45_ind)):
        x = angle_45_ind[i][0]
        y = angle_45_ind[i][1]
        
        # provera lokalnog susedstva za kose ivice pod uglom od 45 stepeni
        if (mag_pad[x,y] > mag_pad[x-1,y+1]) & (mag_pad[x,y] > mag_pad[x+1,y-1]):
            # cuvanje vrednosti ukoliko je ona maksimalna u svom lok. susedstvu
            local_mag_max[x,y] = mag_pad[x,y]
            
    # for petlja za uglove od 90 stepeni
    for i in range(len(angle_90_ind)):
        x = angle_90_ind[i][0]
        y = angle_90_ind[i][1]
        
        # provera lokalnog susedstva za horizontalne ivice
        if (mag_pad[x,y] > mag_pad[x,y-1]) & (mag_pad[x,y] > mag_pad[x,y+1]):
            # cuvanje vrednosti ukoliko je ona maksimalna u svom lok. susedstvu
            local_mag_max[x,y] = mag_pad[x,y]
        
    # for petlja za uglove od -45 stepeni
    for i in range(len(angle_minus_45_ind)):
        x = angle_minus_45_ind[i][0]
        y = angle_minus_45_ind[i][1]
        
        # provera lokalnog susedstva za kose ivice pod uglom od -45 stepeni
        if (mag_pad[x,y] > mag_pad[x-1,y-1]) & (mag_pad[x,y] > mag_pad[x+1,y+1]):
            # cuvanje vrednosti ukoliko je ona maksimalna u svom lok. susedstvu
            local_mag_max[x,y] = mag_pad[x,y]
            
    # vracanje matrice sa lokalno maksimalnim vrednostima magnitude gradijenta 
    # u originalnim dimenzijama
    return local_mag_max[1:M+1, 1:N+1]

def check_eight_neighborhood(img, x, y):
    
    """
    Opis: 
        Funkcija vrsi proveru da li se u 8-susedstvu piksela sa ulazne slike, definisanog
        parametrima/koordinatama x i y, nalazi bilo 1 jedinicni piksel. Provera se vrsi
        sabiranjem svih okolnih piksela. Ukoliko je njihov zbir razlicit od nule, onda u
        8-susedstvu imamo neki jedinicni piksel, u suprotnom u 8-susedstvu nema nijednog 
        jedinicnog piksela.
        
    Parametri:
        img - ulazna slika nad kojom se vrsi provera piksela i njihovog 8-susedstva
        x - x-koordinata piksela u slici img, nad kojim vrsimo proveru 8-susedstva
        y - y-koordinata piksela u slici img, nad kojim vrsimo proveru 8-susedstva
    
    Funkcija vraca vrednost True/False ukoliko se u 8-susedstvu nalazi bilo 1 jedinicni piksel,
    odnosno ne nalazi nijedan jedinicni piksel. 
    
    """
    
    return (img[x-1,y-1] + img[x-1,y] + img[x-1,y+1] + img[x,y-1] + img[x,y+1] + img[x+1,y-1] + img[x+1,y] + img[x+1,y+1]) != 0

def join_weak_edges(strong_edges, weak_edges):
    
    """
    Opis: 
        Funkcija vrsi spajanje slabih i jakih ivica kontinualno
        sve dok se ne dodje do trenutka kada vise nema mogucih 
        spajanja. 
        
    Parametri:
        strong_edges - ulazna mapa jakih ivica
        weak_edges - ulazna mapa slabih ivica
    
    Funkcija vraca matricu/mapu/sliku sa spojenim jakim i slabim ivicama
    
    """
    # dohvatanje dimenzija matrice/mape/slike
    [M,N] = shape(strong_edges)

    # prosirivanje mapa jakih i slabih ivica za po 1 piksel na svaku stranu
    # zbog proveravanja 8-susedstva
    strong = zeros((M+2, N+2))
    weak = zeros((M+2, N+2))
    
    weak[1:M+1, 1:N+1] = weak_edges.copy()
    
    strong[1:M+1, 1:N+1] = strong_edges.copy()
        
    # provera da li uopste ima piksela u mapi slabih ivica
    # iskoriscena je ugradjena funkcija iz biblioteke numpy
    # count_nonzero, koja broji piksele slike koji nisu nula
    if np.count_nonzero(weak) != 0 :
        # promenljiva cnt_joined je brojac koji broji koliko se 
        # piksela iz mape slabih ivica pripojilo mapi jakih ivica
        # ona takodje sluzi i za terminaciju spajanja slabih i jakih ivica
        cnt_joined = 0
        
        # "beskonacna" petlja za spajanje ivica, koja ce se prekinuti
        # u trenutku kada vise nismo uspeli ni jedan piksel da spojimo
        while 1:
            # nalazimo indekse na mestima gde postoje slabe ivice
            # ovako mozemo da prolazimo samo kroz jednu for-petlju 
            # sa manje iteracija, nego li da prolazimo kroz celu matricu 
            # sa dve ugnjezdene for petlje
            weak_ind = np.argwhere(weak) 
            
            # for petlja po pikselima koji pretstavljaju slabe ivice
            for i in range(len(weak_ind)):
                # proverava se 8-susedstvo svakog piksela iz slabih ivica, ali u jakim ivicama
                if check_eight_neighborhood(strong, weak_ind[i][0], weak_ind[i][1]):
                    # ukoliko postoji validno 8-susedstvo, onda se na datoj poziciji
                    # u mapi jakih ivica upisuje 1, a iz mape slabih ivica se brise 
                    # taj piksel
                    strong[weak_ind[i][0], weak_ind[i][1]] = 1
                    weak[weak_ind[i][0], weak_ind[i][1]] = 0
                    # inkrementiramo brojac kao indikaciju da smo napravili promenu
                    cnt_joined += 1
            
            # proveravamo da li je brojac ostao nula
            # ukoliko jeste, iskacemo iz while petlje, jer to znaci 
            # da vise ne mozemo da spojimo slabe ivice sa jakim
            # ukoliko nije, resetujemo brojac i ulazimo u novu iteraciju while petlje
            if cnt_joined == 0:
                break
            else:
                cnt_joined = 0
        
    # vracamo mapu jakih ivica u originalim dimenzijama
    return strong[1:M+1, 1:N+1]

def canny_edge_detection(img_in, sigma, threshold_low, threshold_high, internal_plot = False):
    
    """
    Opis: 
        Funkcija uzima ulaznu sliku i primenjuje Kanijev algoritam 
        za detekciju ivica nad njom. Funkcija postepeno vrsi sledece korake:
            - Niskofrekventno filtriranje slike Gausovim filtrom
            - Odredjivanje horizontalnih i vertikalnih gradijenata filtrirane slike 
              Sobelovim operatorom, kao i odredjivanje magnitude i ugla gradijenta
            - Kvanitizacija ugla gradijenta na pravce 0 dg, 45 dg, 90 dg, -45 dg
            - Potiskivanje lokalnih ne-maksimuma
            - Formiranje mapa jakih i slabih ivica na osnovu ulazih pragova
            - Spajanje mapa slabih i jakih ivica u izlaznu mapu ivica
        
    Parametri:
        img_in - ulazna slika
        sigma - standardna devijacija Gausovom LP filtra
        threshold_low - donji prag za potiskivanje suma / odredjivanje mape slabih ivica
        threshold_high - gornji prag za potiskivanje suma / odredjivanje mape jakih ivica
    
    Funkcija vraca sliku detektovanih ivica. 
    
    """
    img_in_gray = img_in.copy()
    
    if img_in.ndim > 2:
        # pretvaranje slike u grayscale, ukoliko nije u grayscale - u
        img_in_gray = color.rgb2gray(img_in)
    
    # dohvatanje dimenzija ulazne slike
    [M, N] = shape(img_in_gray)
    
    # niskofrekventno filtriranje Gausovim filtrom
    # radius filtra je 3 sigma ---> truncate = 3
    img_in_gauss = filters.gaussian(img_in_gray, sigma, mode='mirror', truncate=3)
    
    # odredjivanje horizontalnog i vertikalnog gradijenta, kao i
    # odredjivanje magnitude i ugla gradijenta
    [Gx, Gy, mag, angle] = sobel_gradient(img_in_gauss)
    
    # kvantizacija gradijenta na pravce 0 deg, 45 deg, 90 deg, -45 deg
    angle_quant = angle_quantization(angle)
    
    # potiskivanje lokalnih ne-maksimuma
    mag_max = local_non_max_suppression(mag, angle_quant)
    
    # odredjivanje mapa jakih i slabih ivica na osnovu zadatih pragova 
    strong_edges = (mag_max >= threshold_high).astype('int')
    weak_edges = ((mag_max >= threshold_low) & (mag_max < threshold_high)).astype('int')
    
    # ukljucivanje slabih ivica u izlazne ivice
    edges = join_weak_edges(strong_edges, weak_edges)
    
    # iscrtavanje rezultata
    if internal_plot:
        fig, ax = plt.subplots(nrows = 6, ncols = 2, figsize = (20, 30), dpi = 120)
        
        ax[0,0].imshow(img_in, cmap = 'jet')
        ax[0,0].set_title('Ulazna slika', fontsize = 16)
        ax[0,0].set_axis_off()
        
        ax[0,1].imshow(img_in_gray, cmap = 'gray')
        ax[0,1].set_title('Ulazna slika - grayscale', fontsize = 16)
        ax[0,1].set_axis_off()
        
        ax[1,0].imshow(img_in_gauss, cmap = 'gray')
        ax[1,0].set_title('Ulazna slika filtrirana Gausovim filtrom \n sigma = ' + str(sigma), fontsize = 16)
        ax[1,0].set_axis_off()
        
        ax[1,1].imshow(Gx, cmap = 'gray', vmin = amin(Gx), vmax = amax(Gx))
        ax[1,1].set_title('Horizontalne ivice - vertikalni gradijent', fontsize = 16)
        ax[1,1].set_axis_off()
        
        ax[2,0].imshow(Gy, cmap = 'gray', vmin = amin(Gy), vmax = amax(Gy))
        ax[2,0].set_title('Vertikalne ivice - horizontalni gradijent', fontsize = 16)
        ax[2,0].set_axis_off()
        
        im0 = ax[2,1].imshow(mag, cmap = 'gray')
        ax[2,1].set_title('Magnituda gradijenta', fontsize = 16)
        ax[2,1].set_axis_off()
        
        im1 = ax[3,0].imshow(angle, cmap = 'jet')
        ax[3,0].set_title('Ugao gradijenta', fontsize = 16)
        ax[3,0].set_axis_off()
        
        im2 = ax[3,1].imshow(angle_quant, cmap='jet')
        ax[3,1].set_title('Kvantizovanu ugao gradijenta', fontsize = 16)
        ax[3,1].set_axis_off()
        
        im3 = ax[4,0].imshow(mag_max, cmap = 'gray')
        ax[4,0].set_title('Maksimalne magnitude gradijenta', fontsize = 16)
        ax[4,0].set_axis_off()
        
        fig.colorbar(im0, ax=ax[2,1], fraction=0.03, pad=0.04)
        fig.colorbar(im1, ax=ax[3,0], fraction=0.03, pad=0.04)
        fig.colorbar(im2, ax=ax[3,1], fraction=0.03, pad=0.04)
        fig.colorbar(im3, ax=ax[4,0], fraction=0.03, pad=0.04)
        
        ax[4,1].imshow(strong_edges, cmap = 'gray')
        ax[4,1].set_title('Mapa jakih ivica \n T_h = ' + str(threshold_high), fontsize = 16)
        ax[4,1].set_axis_off()
        
        ax[5,0].imshow(weak_edges, cmap = 'gray')
        ax[5,0].set_title('Mapa slabih ivica \n T_l = ' + str(threshold_low), fontsize = 16)
        ax[5,0].set_axis_off()
        
        ax[5,1].imshow(edges, cmap = 'gray')
        ax[5,1].set_title('Izlazna mapa ivica', fontsize = 16)
        ax[5,1].set_axis_off()
        
        plt.tight_layout()
        plt.show()
    
    return edges
###########################################################

img = img_as_float(imread(folder_path + 'test2.jpg'))
sigma = 2
threshold_low = 0.2
threshold_high = 0.4

print('Testiranje Kanijevog algoritma na ulaznoj test slici test2.jpg :\n')
print('Parametri funkcije: \n \tsigma = ' + str(sigma) + '\n \tthreshold_low = ' + str(threshold_low) +  '\n \tthreshold_high = ' + str(threshold_high))

edges = canny_edge_detection(img, sigma, threshold_low, threshold_high, True)

###########################################################

img = img_as_float(imread(folder_path + 'test2.jpg'))
sigma = 5
threshold_low = 0.3
threshold_high = 0.8

print('Testiranje Kanijevog algoritma na ulaznoj test slici test2.jpg :\n')
print('Parametri funkcije: \n \tsigma = ' + str(sigma) + '\n \tthreshold_low = ' + str(threshold_low) +  '\n \tthreshold_high = ' + str(threshold_high))

edges = canny_edge_detection(img, sigma, threshold_low, threshold_high, True)
###########################################################

img = img_as_float(imread(folder_path + 'test2.jpg'))
sigma = 2
threshold_low = 0.01
threshold_high = 0.2

print('Testiranje Kanijevog algoritma na ulaznoj test slici test2.jpg :\n')
print('Parametri funkcije: \n \tsigma = ' + str(sigma) + '\n \tthreshold_low = ' + str(threshold_low) +  '\n \tthreshold_high = ' + str(threshold_high))

edges = canny_edge_detection(img, sigma, threshold_low, threshold_high, True)
###########################################################

print('Testiranje Kanijevog algoritma na ulaznim test slikama:')
fig, ax = plt.subplots(nrows = 6, ncols = 3, figsize = (16,20), dpi = 100)
for i in range(1,7,1):
    img_name = 'test'+str(i)+'.jpg'
    img_in = img_as_float(imread(folder_path + img_name))
    start = time()
    img_canny = canny_edge_detection(img_in, 2, 0.2, 0.4)
    end = time()
    time_canny = end-start
    img_canny_skimage = feature.canny(color.rgb2gray(img_in), sigma=2, low_threshold=0.2, high_threshold=0.4)
    
    match = 100*ssim(img_canny, img_canny_skimage, data_range=1)
    
    print('----- slika ' + img_name + ' -----')
    print('Vreme izvrsavanja implementiranog Kanijevog alogritma : ' + "{:.3f}".format(time_canny) + 's')
    print('Poklapanje izmedju implementiranog Kanijevog alogitma i ugradjenog algoritma iznosi: ' + "{:.3f}".format(match) + '%')
    
    ax[i-1,0].imshow(img_in, cmap = 'jet')
    ax[i-1,0].set_title('Ulazna test slika - ' + img_name, fontsize = 16)
    ax[i-1,0].set_axis_off()
    
    ax[i-1,1].imshow(img_canny, cmap = 'gray')
    ax[i-1,1].set_title('Detekcija ivica na slici - ' + img_name, fontsize = 16)
    ax[i-1,1].set_axis_off()
    
    ax[i-1,2].imshow(img_canny_skimage, cmap = 'gray')
    ax[i-1,2].set_title('Detekcija ivica na slici (skimage) - ' + img_name, fontsize = 16)
    ax[i-1,2].set_axis_off()

plt.tight_layout()
plt.show()
###########################################################

print('Testiranje Kanijevog algoritma na segmetisanim ulaznim test slikama:')
fig, ax = plt.subplots(nrows = 6, ncols = 3, figsize = (16,20), dpi = 100)
for i in range(1,7,1):
    img_name = 'test'+str(i)+'.jpg'
    img_in = img_as_float(imread(folder_path + img_name))
    img_seg = segment_lanes(img_in)
    start = time()
    img_canny = canny_edge_detection(img_seg, 2, 0.2, 0.4)
    end = time()
    time_canny = end-start

    print('----- slika ' + img_name + ' -----')
    print('Vreme izvrsavanja implementiranog Kanijevog alogritma : ' + "{:.3f}".format(time_canny) + 's')
    
    ax[i-1,0].imshow(img_in, cmap = 'jet')
    ax[i-1,0].set_title('Ulazna test slika - ' + img_name, fontsize = 16)
    ax[i-1,0].set_axis_off()
    
    ax[i-1,1].imshow(img_seg, cmap = 'gray')
    ax[i-1,1].set_title('Segmentisana slika - ' + img_name, fontsize = 16)
    ax[i-1,1].set_axis_off()
    
    ax[i-1,2].imshow(img_canny, cmap = 'gray')
    ax[i-1,2].set_title('Detekcija ivica na slici - ' + img_name, fontsize = 16)
    ax[i-1,2].set_axis_off()

plt.tight_layout()
plt.show()
###########################################################

def hough_extract(img_edges):
    
    """
    Opis: 
        Funkcija uzima ulaznu mapu ivica i pomocu Hafove
        transformacije odreduje linije na slici. Linije su jednoznacno 
        definisane parametrom theta i rho. Te parametre dobijamo preko hough_line_peaks
        ugradjene funkcije. Ovi parametri se racunaju za levu i desnu traku posebno, jer su 
        uglovi koji se posmatraju razliciti. Pored toga, dominantna je leva
        traka na slici, pa je to jos jedan razlog zasto su ove dve trake
        razdvojene u kodu.
        
    Parametri:
        img_edges - ulazna mapa detektovanih ivica
        
    Funkcija vraca 2 niza od kojih svaki ima po 2 podniza sa parametrima theta i rho, za
    levu i desnu kolovoznu traku posebno.
    
    """
    # leva kolovozna traka
    # opseg uglova koji se posmatra za levu kolovoznu traku
    theta_range = np.linspace(np.radians(45),np.radians(60),1000)
    
    # upotreba ugradjenih funkcija za Hafovu transformaciju i nalazenje 
    # peak-ova Hafove transformacije za levu kolovoznu traku
    [out, angles, distances] = hough_line(img_edges, theta_range)
    [intensity, peak_angles_left, peak_distances_left] = hough_line_peaks(out, angles, distances, min_distance=1, min_angle=0, threshold=amax(out)*0.3, num_peaks=11)
    
    # desna kolovozna traka
    # opseg uglova koji se posmatra za levu kolovoznu traku
    theta_range = np.linspace(np.radians(-60),np.radians(-45),1000)
    
    # upotreba ugradjenih funkcija za Hafovu transformaciju i nalazenje 
    # peak-ova Hafove transformacije za desnu kolovoznu traku 
    # ovaj deo koda se razlikuje samo u parametru min_angle, koji je ovde 20 stepeni 
    # to je upotrebljeno da se ne bi desavala linije ivicama sa strane kolovozne trake - npr. beli auto
    # koji uglavnom ima gotovo vertikalne linije detektovane,
    # e pa bas na ovaj nacin te linije uklanjamo
    [out, angles, distances] = hough_line(img_edges, theta_range)
    [intensity, peak_angles_right, peak_distances_right] = hough_line_peaks(out, angles, distances, min_distance=1, min_angle=20, threshold=amax(out)*0.3, num_peaks=11)
    
    return [[peak_angles_left, peak_distances_left], [peak_angles_right, peak_distances_right]]

def get_one_line_direction(M,N, theta, rho):
    
    """
    Opis: 
        Funkcija uzima ulazne parametre (theta i rho) i na osnovu njih crta liniju
        prostiranja pravca na crnoj slici dimenzija M x N. Dobijena linija je upotrebljena
        kao pokazivac pravca neke detektovane linije na slici. Ova linija koristi se
        dalje u obradi detektovanih linija 
        
    Parametri:
        M, N - dimenzije originalne slike
        theta - parametar koji je vratila Hafova transformacija
        rho - parametar koji je vratila Hafova transformacija
        
    Funkcija vraca sliku sa iscrtanim pravcem prostiranja dobijenim iz
    Hafove transformacije.
    
    """
    # inicijalizacija slike pravca
    img_dir = zeros((M,N))
    
    # provera ukoliko theta i rho nisu nula
    # kad su oba parametra nula, onda program nije detektovao liniju, pa ima prazno 
    # u povratnim parametrima neke od realizovaih funkcija, te je zbog toga ova 
    # provera
    if theta != 0 and rho != 0:
        
        # isrcrtavanje funkcije
        
        # pravljenje vektora horizontalnih vrednosti/indeksa
        hor = np.arange(0, N)
        
        # izracunavanje vertikalnih vrednosti za dobijeni pravac
        ver = (np.round((rho - hor * np.cos(theta)) / np.sin(theta))).astype('int')
        
        # nalazenje indeksa na kojima se nalazi pravac
        ind = (ver >= 0) & (ver < M)
        
        # formirnaje matrice/maske/slike pravca 
        img_dir[ver[ind], hor[ind]] = 1
        
    return img_dir
    
def extract_best_one_line_direction(M, N, hough_line, lane):    
    
    """
    Opis: 
        Funkcija prima izdvojene linije u obliku Hafovih parametara (theta i rho)
        koji se nalaze u ulaznoj listi hough_line i iz njih izvlaci najbolja dva
        pravca za kolovoznu traku (parametar lane definise kol. traku). Najbolji 
        pravac pradstavlja pravac koji ima najduzu liniju na slici dimenzija M x N.
        Ta teorija je opravdana posmatrajuci test slike i video. Linije koje Hafova 
        transformacija i algoritam obrade tih linija koje treba da se izdvoje, 
        odnosno linije kolovoznih traka, su priblizno slicne diagonali slike, pa imaju
        najvecu duzinu. Ova funkcija uzima dva najduza pravca kao dva najbolja reprezenta
        linije kolovozne trake.
        
    Parametri:
        M,N - dimenzije ulazne slike
        hough_line - niz sa podnizovima Hafovih parametara (theta i rho)
        lane - string ('left' ili 'right') kojim se odredjuje koja kol. traka je u pitanju
        
    Funkcija vraca izdvojena dva najbolja pravca (jedan pravac je u formatu [theta,rho])
    za linije jedne kolovozne trake.
    
    """
    
    line = hough_line

    # incijalizacija listi koje se prover
    lengths = []
    thetas = []
    rhos = []
    
    """
    Dodatak:
        Kada su u pitanju desne kolovozne traka, za koje imamo ozbiljan problem
        jer su one bele boje, a kako se segmentacija vrsi na osnovu belih regiona, 
        onda u nekim momentima u frejm/test sliku/sliku upadne i beli auto koji moze
        da napravi pometnju. Ivice se takodje detektuju na njemu, ali i Hafova 
        transformacija detektuje pojedine linije na njemu, iako je min_angle parametar
        funkcije hough_line_peaks, poprilicno dobro to izbegao. 
        Ovom problemu je pristupljeno na sledeci nacin. Dakle, ova funkcija ima za cilj
        da izdvoji najduze linije pravca koji zaklapa neka ivica. Linija je na slikama 
        najduza ako je ta linija zapravo dijagonala slike. S tim razlogom, postavljen je 
        jedan prag (u kodu threshold) koji predstavlja broj piksela za koje duzina linije
        pravca detektovane Hafovom transformacijom moze da odstupa od duzine dijagonale 
        slike. Velicina tog praga odredjena je eksperimentalno, a da pri tome ne dodje do
        devijacije izlaza.
        Ovo poredjenje sa dijagonalom ima smisla, obzirom da su linije kolovoznih traka koje
        treba detektovati i njihovi pravci uglavnom u "blizini dijagonale slike".
    
    """
    
    # broj piksela na dijagonali slike - duzina dijagonale
    diag_pixels_count = np.round(np.sqrt(M**2 + N**2)).astype('int')
    # pomenuti prag za poredjenje sa duzinom dijagonale slike u pikselima
    threshold = 400 #px
    
    # for petlja za prolazak kroz svaki uredjeni par (theta, rho)
    for i in range(len(line[0])):
        # dohvatanje theta
        theta = line[0][i]
        # dohvatanje rho
        rho = line[1][i]
        # definisanje slike pravca za parametre theta i rho 
        # pomocu implementirane fje get_one_line_direction
        img_dir = get_one_line_direction(M, N, theta, rho)
        
        # prebrojavanje jedinicnih piksela u slici pravca - duzina slike formalno
        length = np.count_nonzero(img_dir)
        
        # ako je u pitanju desna kolovozna traka, oda radimo proveru sa pragom
        # ako je duzina manja od razlike duzine dijagonale i praga, onda ta linija otpada
        # ukoliko ne, onda se liste pune sa vrednostima za tu pravu
        if (lane == 'right') and (length >= (diag_pixels_count - threshold)):
            thetas.append(theta)
            rhos.append(rho)
            lengths.append(length)
        # za levu kolovoznu traku, ne vrsimo nikakvu selekciju, zato sto
        # je segmetacija po boji dobro lokalizovana
        elif lane == 'left':
            thetas.append(theta)
            rhos.append(rho)
            lengths.append(length)
            
    # inicijalizacija izlaznih vrednosti
    theta_max = []
    rho_max = []
    
    theta_max2 = []
    rho_max2 = []
    
    # provera da li imamo makar 1 clan u listi
    if len(lengths) > 0:
        # trazimo clan koji ima najvecu duzinu linije pravca
        # njega uzimamo kao dobrog reprezenta
        length_max = max(lengths)
        ind_max = lengths.index(length_max)
        theta_max = thetas[ind_max]
        rho_max = rhos[ind_max]
    
        # brisemo tog clana iz liste, jer
        # nam treba jos jedan clan, sledeci najveci
        del lengths[ind_max]
        del thetas[ind_max]
        del rhos[ind_max]
        
    # ponovo provera da li nakon izbrisanog prvog imamo makar jos
    # jednog clana liste
    if len(lengths) > 0:
        # trazimo opet element liste koji ima najduzu liniju pravca
        length_max2 = max(lengths)
        ind_max2 = lengths.index(length_max2)
        theta_max2 = thetas[ind_max2]
        rho_max2 = rhos[ind_max2]
    
    
    """
    Dodatak:
        Uzimaju se dve najvece vrednosti, jer je eksperimentalno
        odredjeno da uglavnom 3 linije prodju po jednoj kolovonzoj traci
        pa 2 sigurno dobro opisuju tu traku.
    
    """
    
    # vracanje 2 maksimalne vrednosti u formatu njihovih parametara iz Hafovofg prostora
    return [[theta_max, rho_max], [theta_max2, rho_max2]]

def get_line_size(corner_down, corner_up):
    
    """
    Opis: 
        Funkcija racuna celobrojnu vrednost euklidskog rastojanja
        izmedju dve tacke na slici. 
        
    Parametri:
        corner_down - niz od 2 elementa, prvi element je x koordinata tacke A
                      drugi element je y koordinata tacke A
        corner_up - niz od 2 elementa, prvi element je x koordinata tacke B
                      drugi element je y koordinata tacke B
                      
        ****** u opstem slucaju ovo ne mora biti corner_down i corner_up
               vec bilo koje dve tacke
        
    Funkcija vraca celobrojnu vrednost euklidskog rastojanja izmedju dve tacke 
    na slici.
    
    """
    # izvlacanje x koordinata dve tacke
    x1 = corner_down[0]
    x2 = corner_up[0]
    
    # izvlacenje y koordinata dve tacke
    y1 = corner_down[1]
    y2 = corner_up[1]
    
    # racunanje euklidskog rastojanja
    size = np.round(np.sqrt((x1-x2)**2 +(y1-y2)**2)).astype('int')
    
    return size

def get_longest_line_segment(line_segments):
    
    """
    Opis: 
        Funkcija prima duzi, odnosno segmente duzi na nekom pravcu
        dobijenih pomocu funkcije get_line_segments u obliku [x1, y1, x2, y2], 
        gde su (x1, y1) koordinate pocetne tacke duzi/segmenta, a (x2, y2) 
        koordinate krajnje tacke duzi/segmenta.
        
    Parametri:
        line_segments - niz sa podnizovima koji jednoznacno definisu jednu duz
        
    Funkcija vraca onaj line_segment, odnosno onu duz koja je najveca i najbolje 
    reprezentuje sve duzi iz skupa slicnih duzi za pravac jedne kolovozne trake.
    
    """
    
    # prebacivanje u array ulazne liste duzi
    line_segments = np.array(line_segments)
    
    # inicijalizacija najduze duzi i duzine najduze duzi
    longest_line_segment = []
    longest_size = 0

    # provera da li uopste ima duzi koje treba da se obrade
    if len(line_segments) > 0:
        # for petlja kroz sve elemente niz sa podnizovima
        for i in range(line_segments.shape[0]):
            # izvlacenje koordinata krajnjih tacaka duzi za
            # svaki podniz ulaznog niza
            x1 = line_segments[i][0]
            y1 = line_segments[i][1]
            x2 = line_segments[i][2]
            y2 = line_segments[i][3]
            
            # racunanje duzine (euklidsko rastojanje krajnjih 
            # tacaka duzi) duzi i uporedjivanje sa prethodno
            # definisanim maksimumom
            size = get_line_size([x1, y1], [x2, y2])
            if size > longest_size:
                # odredjivanje maksimalne duzi
                longest_size = size
                longest_line_segment = line_segments[i]
    # vracanje samo najduze duzi u formatu [x1, y1, x2, y2]
    return longest_line_segment

def get_line_segments(img_edges, line, min_size, max_gaps, tolerancy):
    
    """
    Opis: 
        Funkcija vrsi izdvajanje duzi minimalne duzine (min_size) sa slike
        img_edges, i to na pravcu definisanim parametrom line (line = [theta, rho]),
        pri cemu dve duzi spaja ukoliko je prosto izmedju njih manji od max_gaps, 
        i pri cemu posmatra samo ivicne piksele koji upadaju u opseg pravac +- tolerancy.
        
    Parametri:
        img_edges - ulazna slika/mapa ivica
        line - niz od dva elementa ([theta i rho]) koji predstavljaju dobijene parametre Hafove trensformacije
        min_size - minimalna velicina duzi koju treba detektovati (u pikselima)\
        max_gaps - maksimalni razmak izmedju dve duzi na istom pravcu koji se tolerise
        tolerancy - radijus oko pravca u kojem trazimo ivicne piksele za duzi
        
    Funkcija vraca niz izdvojenih duzi u formatu [x1, y1, x2, y2], gde uredjeni 
    parovi [x1, y1] i [x2, y2] predstavlju pocetnu i krajnju koordinatu duzi. 
    
    """
    
    # dohvatnje dimenzija ulazne slike
    [M,N] = shape(img_edges)
    
    # inicijalizacija ulaznih parametara iz Hafove transformacije
    theta = 0
    rho = 0
    # vrsi se provera postojanja ovih parametara
    # posto se oni ne prosledjuju direktno iz Hafove transformacije
    if len(line) > 0:
        # dohvatanje parametara theta i rho
        theta = line[0]
        rho = line[1]
    
    """
        Ideja je da se na osnovu dobijenih parametar theta i rho 
        iscrta linija pravca koju definisu ta dva parametra. Nakon toga 
        ta linija da se prosiri dilatacijom sa maksom u obliku diska i radijusom
        velicine tolerancy i da se tako dobijen slika "enduje" odnosno pomnozi sa
        slikom ivica. Na taj nacin smo dobili sliku koja ima piksele samo 
        po trazenom pravcu i to sa njegovom okolinom u obliku +- tolerancy.
        Ovako ne moramo za svaki piksel pojedinacno da proveravamo susedstvo na osnovu
        parametra tolerancy. Ovako dobijena maskirana slika ide dalje na 
        pretragu segmenata.
    """
    
    # formiranje slike linije pravca na osnovu zadatih parametara theta i rho
    img_dir = get_one_line_direction(M, N, theta, rho)
    
    # dobijena slika pravca se obradjuje
    # slika se dilatira, odnosno prosiruje binarnom dilatacijom sa maskom u obliku
    # diska ciji je radijus upravo parametar tolerancy
    img_dir_tol = morphology.binary_dilation(img_dir, morphology.disk(tolerancy))
    img_extracted = img_edges*img_dir_tol
    
    """
        Ideja je da se krene od dna dobijene slike sa izdvojenim validnim
        ivicnim pikselima, po vrstama i da se ukoliko naidje na jedinicni piksel
        u promenljivu arr cuva njegova pozicija. Potom posmatramo sve bele piksele
        i njihove indekse koji se nalaze u nizu arr. Posmatra se razlika sva susedna
        podniza u nizu arr i to po x koordinati, dakle gledamo razliku x koordinata 
        u kojima se nalaze beli pikseli. Ako je ta razlika tacno 1, znaci da u dve
        susedne vrste imamo bele piksele, pa to sve zajedno spajamo u jednu duz.
        Ukoliko ta razlika nije 1, vrsimo proveru sa parametrom max_gaps. Dakle, 
        ako je razlika manja od max_gaps, ta dva regiona cemo spojiti, ukoliko nije manja
        Onda imamo dve duzi. Naravno, uvek prilikom fomriranja finalne duzi, vrsimo proveru
        njene duzine i uporedjujemo je sa min_size parametrom. Pri tome, ne moramo 
        da se brinemo za razliku po y osi, jer smo to sve ogranicili sa tolerancy, prilikom 
        "endovanja".
    """
    
    # inicijalizacija listi za cuvanje podataka
    segments = [] # lista za sve duzi

    arr = [] # lista za indekse jedinicnih piksela
    down_corner = [] # lista za donje / pocetne tacke duzi
    up_corner = [] # lista za gornje / krajnje tacke duzi

    # for petlja za prolazak kroz maskiranu sliku od dna po vrstama
    for i in range(M-1,-1,-1):
        # nalazenje jedinicih piksela, fja argwhere vraca indeks niza
        ind = np.argwhere(img_extracted[i,:] == 1)
        if ind.size > 0:
            # dodavanje indeksa u niz svih jedinicnih piksela na slici
            arr.append([i, ind])
    # konverzija u np.array
    arr = np.array(arr, dtype = 'object') 
    # postavljanje donje /pocetne tacke duzi
    # dakle uzimamo najvece x i najvece y iz niza arr
    # najvece x -> najniza tacka na slici, 
    # najvece y -> najlevlja tacka na pravcu, ali u vrsti x
    down_corner  = np.array([arr[0,0], max(arr[0,1])[0]])
    
    # prolazimo kroz sve bele piksele sa slike i posmatramo njihove indekse
    for i in range(arr.shape[0]-1):
        # racunamo razliku izmedju dva susedna clana u nizu arr i to po prvoj koordinati
        # ta razlika predstavlja razliku rednih brojeva vrste kojoj beli piksel pripada
        diff = arr[i,0] - arr[i+1,0]
        # provera da li je razlika u intervalu [1, max_gaps], ako jeste
        # prepisujemo u gornju/krajnju tacku duzi sledecu vrstu i max po y i nastavljamo dalje
        if diff >= 1 and diff <= max_gaps:
            up_corner = np.array([arr[i+1,0], max(arr[i,1])[0]])
        # ukoliko je razlika veca od max_gaps
        elif diff > max_gaps:
            # za krajnju tacku te duzi uzimamo bas ovu vrstu i max po y
            up_corner = np.array([arr[i,0], max(arr[i,1])[0]])
            
            # potom proveravamo da li je duzina linije veca od min_size
            size =  get_line_size(down_corner, up_corner)
            if size >= min_size:
                # ukoliko jeste, onda u promenljivu segments 
                # stavljamo ovaj niz vrednosti pocetnih i krajnih tacaka
                segments.append([down_corner, up_corner])

            # u svakom slucaju, nebitno od duzine, donju/pocetnu tacku nove
            # duzi stavljamu u sledecu vrstu i max po y
            down_corner =  np.array([arr[i+1,0], max(arr[i+1,1])[0]])      

    # na kraju ostaje poslednju duz da proverimo nakon sto smo izasli iz
    # for petlje , te ovde proveravamo njenu duzinu
    size =  get_line_size(down_corner, up_corner)
    if size >= min_size:
        segments.append([down_corner, up_corner])

    # konverzija u np.array
    segments = np.array(segments, dtype = 'object')
    
    # finalno pakovanje rezultata
    line_segments = []
    
    # pakovanje rezulata u formatu koji je potreban za 4. zadatak
    for i in range(segments.shape[0]):
        line_segments.append([segments[i][0][0], segments[i][0][1], segments[i][1][0], segments[i][1][1]])
    
    return line_segments
###########################################################

def lane_detection(video_frame, internal_plot = False):
    
    """
    Opis: 
        Funkcija uzima ulaznu sliku, vrsi segmentaciju kolovoznih traka na slici,
        potom primenom Kanijevog algoritma vrsi detekciju ivica na slici, izvlaci linije
        sa slike primenom Hafove transformacije, vrsi izvlacenje duzi na definisanim 
        ivicama i pravcima prostiranja ivica i konacno uzima najvece duzi za levu i 
        desnu kolovoznu traku koje prosledjuje kao rezultat.
        
    Parametri:
        video_frame - ulazni frejm/slika koja se obradjuje
        
    Funkcija vraca niz sa dva podniza u formatu [[xl1,yl1,xl2,yl2],[xr1,yr1,xr2,yr2]], 
    gde su podnizovi zapravo koordinate duzi za levu i desnu kolovoznu traku.
    
    """
    # cuvanje ulazne slike, za potrebe prikaza rezultata
    img_original = video_frame.copy()
    
    # segmeentacija ulazne slike - izdvajanje regiona od interesa
    img_segmentated = segment_lanes(video_frame, False)
    
    # parametri za Kanijev algoritam, najbolji
    sigma = 2
    threshold_low = 0.2
    threshold_high = 0.4
    
    # formiranje mape ivica ulazne slike primenom Kanijevog algoritma
    img_edges = canny_edge_detection(img_segmentated, sigma, threshold_low, threshold_high)
    
    # izlvacenje dimenzija slike u grayscale obliku
    [M,N] = shape(img_edges)
    
    # ekstrakcija linija pomocu Hafove transformacije
    [left_lane_hough_params, right_lane_hough_params] = hough_extract(img_edges)
    
    # izdvajanje najbolje 2 linije od svih ekstraktovanih Hafovom transformacijom za svaku traku
    [left_lane_best_direction_1, left_lane_best_direction_2] = extract_best_one_line_direction(M, N, left_lane_hough_params, 'left')
    [right_lane_best_direction_1, right_lane_best_direction_2] = extract_best_one_line_direction(M, N, right_lane_hough_params, 'right')
    
    # parametri za izdvajanje duzi
    min_size = 10
    max_gaps = 5
    tolerancy = 12
    
    # inicijalizacija listi za izdvojene duzi, za sve 4 vrednosti najpovoljnih Hafovih parametara
    # (po dve za svaku kolovoznu traku)
    line_segments_left_1 = []
    line_segments_left_2 = []
    line_segments_right_1 = []
    line_segments_right_2 = []
    
    # liste punimo za svaki od 4 najbolje izdvojenih parametara Hafove transformacije
        
    # provera uslova, ukoliko na frejmu ne nadje ni jedna adekvatna linija
    if len(left_lane_best_direction_1) > 0 and (isinstance(left_lane_best_direction_1[0], list) == False):
        # ukoliko postoje dobre linije, ubacijemo ih u listu za tu traku
        line_segments_left_1 = get_line_segments(img_edges, left_lane_best_direction_1, min_size, max_gaps, tolerancy)
        
    # provera uslova, ukoliko na frejmu ne nadje ni jedna adekvatna linija
    if len(left_lane_best_direction_2) > 0 and (isinstance(left_lane_best_direction_2[0], list) == False):
        # ukoliko postoje dobre linije, ubacijemo ih u listu za tu traku
        line_segments_left_2 = get_line_segments(img_edges, left_lane_best_direction_2, min_size, max_gaps, tolerancy)
    
    # spajanje listi za levu kolovoznu traku
    line_segments_left = line_segments_left_1 + line_segments_left_2
    
    # provera uslova, ukoliko na frejmu ne nadje ni jedna adekvatna linija
    if len(right_lane_best_direction_1) > 0 and (isinstance(right_lane_best_direction_1[0], list) == False):    
        # ukoliko postoje dobre linije, ubacijemo ih u listu za tu traku
        line_segments_right_1 = get_line_segments(img_edges, right_lane_best_direction_1, min_size, max_gaps, tolerancy)
        
    # provera uslova, ukoliko na frejmu ne nadje ni jedna adekvatna linija
    if len(right_lane_best_direction_2) > 0 and (isinstance(right_lane_best_direction_2[0], list) == False):
        # ukoliko postoje dobre linije, ubacijemo ih u listu za tu traku
        line_segments_right_2 = get_line_segments(img_edges, right_lane_best_direction_2, min_size, max_gaps, tolerancy)
    
    # spajanje listi za desnu kolovoznu traku
    line_segments_right = line_segments_right_1 + line_segments_right_2

    # na kraju, od svih duzi za levu i desnu kolovoznu traku
    # izdvajamo samo one koje su najduze i vracamo ih kao povratne vrednosti funkcije
    left = get_longest_line_segment(line_segments_left)
    right = get_longest_line_segment(line_segments_right)
    
    # crtanje izvucenih duzi po ulaznoj slici
    if len(left) > 0:
        # iscrtavanje se vrsi u obrnutim koordinatama, jer to radi jedna druga (OpenCV) biblioteka
        video_frame = cv2.line(video_frame,(left[1],left[0]),(left[3],left[2]),(1,0,0),20)
    if len(right) > 0:
        video_frame = cv2.line(video_frame,(right[1],right[0]),(right[3],right[2]),(1,0,0),20)
    
    if internal_plot:
        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (20,16), dpi = 100)
            
        ax[0,0].imshow(img_original, cmap = 'jet')
        ax[0,0].set_title('Ulazna slika', fontsize = 16)
        ax[0,0].set_axis_off()
        
        ax[0,1].imshow(img_segmentated, cmap = 'gray')
        ax[0,1].set_title('Segmentisana slika', fontsize = 16)
        ax[0,1].set_axis_off()
        
        ax[1,0].imshow(img_edges, cmap = 'gray')
        ax[1,0].set_title('Detektovane ivice na slici', fontsize = 16)
        ax[1,0].set_axis_off()
        
        ax[1,1].imshow(video_frame, cmap = 'jet')
        ax[1,1].set_title('Izlazna slika', fontsize = 16)
        ax[1,1].set_axis_off()
        
        plt.tight_layout()
        plt.show()
    
    return left, right
###########################################################

print('Prikaz rezultata sa medjukoracima funkcije lane_detection na ulaznim test slikama :')
for i in range(1,7,1):
    img_name = 'test'+str(i)+'.jpg'
    
    print('----- slika ' + img_name + ' -----')
    
    img_in = img_as_float(imread(folder_path + img_name))
    start = time()
    left, right = lane_detection(img_in, True)
    end = time()
    time_lane = end-start
    
    print('Vreme izvrsavanja implementirane funkcije lane_detection za trenutnu sliku : ' + "{:.3f}".format(time_lane) + 's')
###########################################################

print('Konacni prikaz rezultata implementirane funkcije lane_detection za ulazne test slike:')
fig, ax = plt.subplots(nrows = 6, ncols = 2, figsize = (20,30), dpi = 100)
for i in range(1,7,1):
    img_name = 'test'+str(i)+'.jpg'
    img_in = img_as_float(imread(folder_path + img_name))
    img_original = img_in.copy()
    start = time()
    left, right = lane_detection(img_in)
    end = time()
    time_lane = end-start
    
    # iscrtavanje linija na ulaznim test slikama, 
    # linije predstavljaju detektovane kolovozne trake
    if len(left) > 0:
        # iscrtavanje se vrsi u obrnutim koordinatama, jer to radi jedna druga (OpenCV) biblioteka
        img_in = cv2.line(img_in,(left[1],left[0]),(left[3],left[2]),(1,0,0),20)
    if len(right) > 0:
        img_in = cv2.line(img_in,(right[1],right[0]),(right[3],right[2]),(1,0,0),20)

    print('----- slika ' + img_name + ' -----')
    print('Vreme izvrsavanja implementirane funkcije lane_detection za trenutnu sliku : ' + "{:.3f}".format(time_lane) + 's')
    
    ax[i-1,0].imshow(img_original, cmap = 'jet')
    ax[i-1,0].set_title('Ulazna test slika - ' + img_name, fontsize = 16)
    ax[i-1,0].set_axis_off()
    
    ax[i-1,1].imshow(img_in, cmap = 'jet')
    ax[i-1,1].set_title('Ulazna test slika sa detektovanim kolovoznim trakama - ' + img_name, fontsize = 16)
    ax[i-1,1].set_axis_off()

plt.tight_layout()
plt.show()
###########################################################

def get_frames(video_name, video_extenstion):
    
    """
    Opis: 
        Funkcija ucitava video specificiran nazivom i exktenzijom, 
        izdvaja ulazne frejmove iz njega i cuva ih u specificiran folder.
        
    Parametri:
        video_name - naziv videa
        video_extenstion - format/extenzija videa (npr. mp4)
        
    Funkcija vraca broj frejmova/slika koje se nalaze u videu i broj
    fps - frames per second i pored toga generise folder za cuvanje
    izvucenih frejmova i samo frejmove.
    
    """
    
    full_video_name = video_name + '.' + video_extenstion
    
    # naziv foldera za smestanje izvucenih frejmova
    frame_path = video_name + "_frames"
    
    # kreiranje foldera na racunaru na specificiranoj putanji
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    
    # kreiranje objekta videa koji se nalazi u folderu sekvence
    video_capture = cv2.VideoCapture( folder_path + full_video_name)
    
    # dohvatanje fps za ulazni video
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # ucitavanje prvog frejma
    success, frame = video_capture.read()
    # resetovanje brojaca frejmova
    cnt = 0 
    # nakon ovoga ulazimo u while petlju
    # koja se prekida onda kada nema vise frejmova za
    # citanje
    while success:
        # upisivanje novog frejma na putanju 
        cv2.imwrite(frame_path + "/frame%d.jpg" % cnt, frame)
        
        # sledeci frejm
        success, frame = video_capture.read()
        
        cnt += 1
    
    return [cnt, fps]

def draw_detected_lanes(video_name, frame_number):  
    
    """
    Opis: 
        Funkcija ucitava sliku iz video sekvence specificirane nazivom,
        zatim se za tu sliku poziva fja lane_detection i preko te slike
        se iscrtavaju linije detektovanih kolovoznih traka.
        
    Parametri:
        video_name - naziv videa
        frame_number - redni broj frejma sacuvanog u folderu
        
    Funkcija nema povratnih vrednosti, medjutim generise se folder
    sa frejmovima sa detektovanim kolovoznim trakama i u folder se ubacuje
    slika/frejm sa detektovanim trakama.
    
    """

    # ime foldera u kojem se cuvaju rezultati
    lanes_detected_folder = video_name + '_lanes_detected'
    
    # kreiranje foldera rezultata
    if not os.path.exists(lanes_detected_folder):
        os.makedirs(lanes_detected_folder)
    
    # folder u kojem se nalazi frejm za obradu
    frame_path = video_name + '_frames/'
    
    # folder u koji se smesta rezultat
    lanes_detected_folder_path = video_name + '_lanes_detected/'
    
    # ucitavanje specificiranog frejma
    image = img_as_float(imread(frame_path + 'frame' + str(frame_number) + '.jpg'))
    
    # obrada slike sa funkcijom lane_detection
    left, right = lane_detection(image)
    
    # iscrtavanje linija po slici
    # obrnute su x i y koordinate
    if len(left) > 0:
        image = cv2.line(image,(left[1],left[0]),(left[3],left[2]),(1,0,0),20)
    if len(right) > 0:
        image = cv2.line(image,(right[1],right[0]),(right[3],right[2]),(1,0,0),20)
    
    # konvertovanje slike za cuvanje
    image = image.astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # cuvanje slike dobije obradom
    cv2.imwrite(lanes_detected_folder_path + 'frame' + str(frame_number) + '.jpg', 255*image)
    
    
def  create_and_store_video(video_name, num_frames, fps):

    """
    Opis: 
        Funkcija kreira video sa slikama obradjenim
        funkcijom lane_detection.
        
    Parametri:
        video_name - naziv videa
        num_frames - ukupan broj frejmova u videu
        fps - fps videa
        
    Funkcija nema povratnih vrednosti, medjutim generise se video 
    koji predstavlja obradjenu video sekvencu.
    
    """
    
    # naziv putanje u kojem se nalaze obradjeni frejmovi
    lanes_detected_folder_path = video_name + "_lanes_detected/"
    
    out = None
    for i in range(0, Nframes):

        # ucitavanje slike za upis u video
        image = cv2.imread(lanes_detected_folder_path + 'frame' + str(i) + '.jpg')

        # kreiranje objekta, ako vec nije kreiran
        if not out:
            # dohvatanje dimenzija slike
            [M, N, D] = shape(image)
            
            # kreiranje objekta za video 
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_name + "_lanes_detected.mp4", fourcc, fps, (N, M))

        # upis slike u video
        out.write(image)

    # kraj upisvanje slika u video
    out and out.release()
    cv2.destroyAllWindows()
    

def lane_detection_from_video(video_name, video_extenstion):
    
    """
    Opis: 
        Funkcija ucitava video specificiran nazivom i exktenzijom, 
        izdvaja ulazne frejmove iz njega, zatim poziva funkciju lane_detection
        i crta detektovane kolovozne trake i na kraju cuva video na odredjenoj
        lokaciji.
        
    Parametri:
        video_name - naziv videa
        video_extenstion - format/extenzija videa (npr. mp4)
        
    Funkcija nema povratnih vrednosti, medjutim generisu se folderi 
    sa originalnim frejmovima, frejmovima sa detektovanim kolovoznim trakama i
    sam izlazni video.
    
    """
    
    # izdvajanje frejmova ulaznog videa
    [num_frames, fps] = get_frames(video_name, video_extenstion)
    
    # obrada frejmova sa funkcijom lane_detection
    for i in range(0, num_frames):
        draw_detected_lanes(video_name, i)
            
    # kreiranje novog videa sa ucrtanim linijama koje 
    # predstavljaju detektovane kolovozne trake
    create_and_store_video(video_name, num_frames, fps)
###########################################################

lane_detection_from_video('video_road','mp4')
###########################################################
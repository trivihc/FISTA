import numpy as np
import datetime
from scipy.signal import butter, sosfilt, spectrogram

from math import ceil
from os.path import basename
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import antropy as ant
from scipy.signal import periodogram, welch, lfilter , decimate
import librosa.display
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert, fftconvolve
import collections
from scipy.signal import find_peaks
from matplotlib.widgets import RectangleSelector
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks
import plotly.colors

from HDAS_File_Open import Load_2D_Data_bin
# from features.base import logfbank 
# from features.base import delta


global window_duration
global overlap_duration

window_duration = 4.0
overlap_duration = 3.8  # shift_duration

# Especifica los puntos espaciales inicial y final que deseas seleccionar
global inicio_punto_espacial
global fin_punto_espacial

# inicio_punto_espacial = 127  # Punto espacial inicial
# fin_punto_espacial = 148  # Punto espacial final (exclusivo) 150 - 117
inicio_punto_espacial = 0 # Punto espacial inicial
fin_punto_espacial = 75
global height_ip
global distance_ip
global prominence_ip
global offset
global  offset_ip

height_ip = 350
distance_ip = 20
prominence_ip = 20
offset = 2000
offset_ip = 10000


class HDAS:

    
    def __init__(self, dir, filenames, facx=-1, dectype='simple', verbose=False):
        # Read data from Aragon Photonics format
        # dir: base directory of the files
        # filenames: list of individual file names
        # facx: spatial decimation factor (-1=no decimation)
        # dectype: type of spatial decimation (if used):
        #          simple: stack between adjacent traces
        #          median: median between adjacent traces
        # verbose: True for debugging purposes
        

        # Read metadata
        [dd, header] = Load_2D_Data_bin("%s/%s" % (dir, filenames[0]))

        self.nsens = dd.shape[0]
        self.dx = header[1]
        self.srate = header[6]/header[15]/header[98]
        self.dt = 1. / self.srate

        # 012345678901234567890123456789
        # 2021_10_19_18h07m05s_HDAS_2Dmap_Strain.bin
        ff = basename(filenames[0])
        YY = int(ff[0:4])
        MM = int(ff[5:7])
        DD = int(ff[8:10])
        HH = int(ff[11:13])
        MI = int(ff[14:16])
        SS = int(ff[17:19])

        self.stime = datetime.datetime(YY,MM,DD,HH,MI,SS)

        self.fmax = self.srate / 2
        self.fmin = 0

        self.verbose = verbose

        # Merge files
        mats = []
        for i in range(len(filenames)):
            if verbose:
                print("Reading ", filenames[i])

            [dd, header] = Load_2D_Data_bin("%s/%s" % (dir, filenames[i]))
            dd = np.nan_to_num(dd)

            if facx > 0:
                dd = self._decimateX(dd, facx, dectype)

            mats.append(dd)

        self.da = np.hstack(tuple(mats))
        self.da = self.da.astype('float32')

        if facx > 0:
            self.dx = self.dx * facx
            self.nsens = self.da.shape[0]

        self.xpos = np.arange(self.nsens) * self.dx

        self.nsamp = self.da.shape[1]
        self.trel = np.arange(self.nsamp) * self.dt
        self.tabs = [self.stime + datetime.timedelta(milliseconds=int(i * self.dt * 1000)) for i in range(self.nsamp)]
        self.etime = self.stime + datetime.timedelta(seconds=self.nsamp * self.dt)
        
        

        if verbose:
            print("Original size (x,t) : ", self.da.shape)

    def _decimateX(self, dd, facx, dectype):
        # Function for internal use only
        self.tabs = [self.stime + datetime.timedelta(milliseconds=int(i * self.dt * 1000)) for i in range(self.nsamp)]

        if dectype == 'simple':
            return dd[::facx, :]

        elif dectype == 'median':
            newsens = int(self.nsens / facx)
            newdd = np.zeros((newsens, self.nsamp), dtype='float32')
            for i in range(newsens):
                ss = dd[i * facx:(i + 1) * facx, :]
                newdd[i, :] = np.median(ss, 0)

            return newdd

    def cutX(self, x1, x2):
        # Cut a spatial range
        # x1,x2: lower and upper limits (in m) of the distance along the cable

        i1 = int((x1 / self.dx))
        i2 = int((x2 / self.dx))
        self.da = self.da[i1:i2, :]
        self.nsens = self.da.shape[0]

        self.xpos = x1 + np.arange(self.nsens) * self.dx

        if self.verbose:
            print("Cut along X (x,t) : ", self.da.shape)

    def decimateX(self, facx, dectype):
        # Spatial decimation
        # facx: spatial decimation factor (-1=no decimation)
        # dectype: type of spatial decimation (if used):
        #          simple: stack between adjacent traces
        #          median: median between adjacent traces

        if dectype == 'simple:':
            self.da = self.da[::facx, :]
            self.xpos = self.xpos[::facx]

        elif dectype == 'median':
            newsens = int(self.nsens / facx)
            newda = np.zeros((newsens, self.nsamp), dtype='float32')
            for i in range(newsens):
                ss = self.da[i * facx:(i + 1) * facx, :]
                newda[i, :] = np.median(ss, 0)

            self.da = newda

        self.dx = self.dx * facx
        self.nsens = self.da.shape[0]

        if dectype == 'simple':
            self.xpos = self.xpos[0] + np.arange(self.nsens) * self.dx
        elif dectype == 'median':
            self.xpos = self.xpos[0] + np.arange(self.nsens) * self.dx + (self.dx * facx / 2)

        if self.verbose:
            print("Decimate along X (x,t) : ", self.da.shape)

    def removeTrend(self):
        # Remove linear trend along individual traces through LSQR fit

        if self.verbose:
            print("Remove trend")

        xx = np.arange(self.nsamp)
        for i in range(self.da.shape[0]):
            dr = self.da[i, :]
            try:
                po = np.polyfit(xx, dr, 1)
            except:
                print(dr)
                #plt.plot(xx,dr,'k-')
                #plt.show()
                exit(1)
            mo = np.polyval(po, xx)
            self.da[i, :] = dr - mo

    def removeCoherentNoise(self, method='median'):
        # Remove coherent synchronous noise from all the traces
        # The trace used for filtering is computed by doing the median
        # of all the individual traces
        #
        # method: filtering type:
        #         simple: subtracts the median from individual traces
        #         fit: compute the best fit correction factor befor subtracting the median

        if self.verbose:
            print("Remove coherent noise")

        md = np.median(self.da, 0)

        if method == 'simple':
            for i in range(self.da.shape[1]):
                self.da[:, i] = self.da[:, i] - md[i]

        elif method == 'fit':

            den = np.sum(md * md)
            for i in range(self.da.shape[0]):
                dd = self.da[i, :]
                am = np.sum(dd * md) / den
                self.da[i, :] = dd - am * md

    def cutT(self, t1, t2):
        # Cut a temporal range
        # t1,t2: lower and upper limits (in s) of the time window to cut

        j1 = int(t1 / self.dt)
        j2 = int(t2 / self.dt)
        self.da = self.da[:, j1:j2]

        self.nsamp = self.da.shape[1]

        self.stime = self.stime + datetime.timedelta(seconds=t1)
        self.etime = self.stime + datetime.timedelta(seconds=self.nsamp * self.dt)

        self.trel = np.arange(self.nsamp) * self.dt
        self.tabs = [self.stime + datetime.timedelta(seconds=i * self.dt) for i in range(self.nsamp)]

        if self.verbose:
            print("Cut along T (x,t) : ", self.da.shape)

    
    def len(self):
        # Returns the temporal length of the traces

        return self.dt*self.nsamp

    def decimateT(self, fact):
        # Temporal decimation
        # fact: decimation factor
        # !!! low-pass filtering below Nyquist is not perfmed here

        self.da = self.da[:, ::fact]
        self.dt = self.dt * fact
        self.srate = self.srate / fact

        self.nsamp = self.da.shape[1]

        self.etime = self.stime + datetime.timedelta(seconds=self.nsamp * self.dt)

        self.trel = np.arange(self.nsamp) * self.dt
        self.tabs = [self.stime + datetime.timedelta(seconds=i * self.dt) for i in range(self.nsamp)]

        if self.verbose:
            print("Decimate along T (x,t) : ", self.da.shape)
            print("Fmax ", self.fmax, " New Nyquist  ", self.srate / 2)

    def filter(self, f1, f2):
        # Band-pass filtering of individual traces
        # Butterworth filter with 4 poles is used
        # f1,f2: lower and upper frequency of the filter

        if self.verbose:
            print("Filter")

        sos = butter(4, [f1, f2], 'bandpass', fs=1. / self.dt, output='sos')
        
        for i in range(self.da.shape[0]):
            dd = self.da[i, :]
            dd = sosfilt(sos, dd)
            self.da[i, :] = dd
            print(f"paso por aqui {i+1} de {self.da.shape[0]}")
        self.fmax = f2
        self.fmin = f1

    def normalize(self, type='rms_c'):
        # Trace normalization
        # type, type of normalization
        #       rms: normalization of the whole DAS image by its RMS
        #       rms_c: normalization of individual traces by their RMS
        #       mad: normalization of the whole DAS image by its MAD (Median Absolute Deviation)
        #       mad_c: normalization of individual traces by their MAD

        if self.verbose:
            print("Normalize ", type)

        if type == 'rms':
            self.da = self.da / np.std(self.da)

        elif type == 'rms_c':
            rms = np.std(self.da, 1)
            for i in range(len(rms)):
                self.da[i, :] = self.da[i, :] / rms[i]

        elif type == 'mad':
            mad = np.median(np.abs(self.da))
            self.da = 0.5 * self.da / mad

        elif type == 'mad_c':
            for i in range(self.nsens):
                mad = np.median(np.abs(self.da[i, :]))
                self.da[i, :] = 0.5 * self.da[i, :] / mad

    def mute(self, perc=95):
        # Muting of noisy traces based on their RMS
        # perc: percentile of RMS above which mute (set to zero) a trace
        # 0 means all the traces muted, 100 no trace is muted

        st = np.std(self.da, axis=1)
        thr = np.percentile(st,perc)
        idx = (st>=thr)
        self.da[idx,:] = 0.

    def check(self):
        # Check the object for debugging purposes
        # writing some of the relevant paramaeters

        print(">>> CHECK HFD5DAS")
        print("NSENS ", self.nsens)
        print("NSAMP ", self.nsamp)
        print("shape ", self.da.shape)
        print("dx,len ", self.dx, self.dx * self.nsens)
        print("xpos ", self.xpos[0], self.xpos[-1])
        print("dt,len ", self.dt, self.dt * self.nsamp)
        print("srate ", self.srate)
        print("stime ", self.stime)
        print("etime ", self.etime)
        print("len ", self.etime - self.stime)
        print("trel ", self.trel[0], self.trel[-1])
        print("tabs ", self.tabs[0], self.tabs[-1])
        print("fmax ", self.fmax)

    def checkHTML(self):
        # Same as check but in HTML format

        print(">>> CHECK HFD5DAS</br>")
        print("NSENS ", self.nsens,"</br>")
        print("NSAMP ", self.nsamp,"</br>")
        print("shape ", self.da.shape,"</br>")
        print("dx,len ", self.dx, self.dx * self.nsens,"</br>")
        print("xpos ", self.xpos[0], self.xpos[-1],"</br>")
        print("dt,len ", self.dt, self.dt * self.nsamp,"</br>")
        print("srate ", self.srate,"</br>")
        print("stime ", self.stime,"</br>")
        print("etime ", self.etime,"</br>")
        print("len ", self.etime - self.stime,"</br>")
        print("trel ", self.trel[0], self.trel[-1],"</br>")
        print("tabs ", self.tabs[0], self.tabs[-1],"</br>")
        print("fmax ", self.fmax,"</br>")
        
        
        
    
    def calculate_instant_power(self, sp):
        """
        Calcula la potencia instantánea sobre todos los datos de un punto espacial.
        """
        # Obtener los datos completos del punto espacial
        data = self.da[sp, :]
        
        # Derivar la señal usando interpolación cúbica y filtrado
        dt = 1 / self.srate  # tiempo entre muestras en la señal original
        dti = 1 / (4 * self.srate)  # intervalo de tiempo tras interpolación
        N = len(data)
        t = np.arange(0, N) * dt  # tiempos originales
        ti = np.arange(0, t[-1] + dti, dti)  # tiempos tras interpolación
    
        # Interpolación cúbica de la señal
        cs = CubicSpline(t, data)
        xi = cs(ti)  # señal interpolada
    
        # Derivación usando un filtro
        b = np.array([1, -1]) / dti  # coeficientes del filtro para derivada
        vi = lfilter(b, 1, xi)  # señal derivada
        v = decimate(vi, 4)  # reducir la frecuencia de muestreo a la original
    
        # Calcular la potencia instantánea como el producto de la señal original y su derivada
        self.instant_power_vals = data * v
    
        # Asegurarse de que el resultado esté en formato array de numpy
        self.instant_power_vals = np.array(self.instant_power_vals)
        
        

    def calculate_instant_power_plot(self):
        """
        Procesa la potencia instantánea para un rango de puntos espaciales y genera un gráfico.
        """

        
        # Inicializa una lista para almacenar resultados de cada punto espacial
        resultados3 = []
        
        # Itera sobre los puntos espaciales seleccionados
        for i in range(inicio_punto_espacial, fin_punto_espacial):
            # Llama a la función para calcular el instant power del punto espacial i
            self.calculate_instant_power(sp=i)
            # Agrega los resultados a la lista
            resultados3.append(self.instant_power_vals)
        
        # Convierte la lista a un array NumPy para obtener un formato de columnas
        self.instant_power_vals_array = np.array(resultados3)
        
        # Invertir el orden de las filas
        self.instant_power_vals_array = self.instant_power_vals_array[::-1]


        # Acceder al valor desde el diccionario
        time_samples = np.arange(self.instant_power_vals_array.shape[1]) 
        
        # Convert specific time to seconds
        time_marker = 7000 # 11:43:24 (primer coche)
        

        # Crear la figura y el eje
        plt.figure(figsize=(12, 8))

        # Graficar cada fila de instant_power_vals_array como líneas continuas
        for i in range(self.instant_power_vals_array.shape[0]):  # 33 filas (invertidas)
            # Desplazar la señal en el eje Y
            plt.plot(time_samples, self.instant_power_vals_array[i] + i * offset_ip, label=f'Señal {self.instant_power_vals_array.shape[0] - i}', linestyle='-', alpha=0.7)
        
        plt.axvline(x=time_samples[time_marker], color='red', linestyle='--', label='Coche 1')
        # Personalizar el gráfico
        plt.title('33 Señales de Instant Power')
        plt.xlabel('Muestras de Tiempo')
        plt.ylabel('Puntos Espaciales')
        plt.yticks(
            np.arange(0, self.instant_power_vals_array.shape[0] * offset_ip, offset_ip), 
            labels=[f'Punto {self.instant_power_vals_array.shape[0] - i}' for i in range(self.instant_power_vals_array.shape[0])]
        )  # Etiquetas invertidas para los puntos espaciales

        plt.grid()
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Mueve la leyenda fuera del gráfico
        plt.tight_layout()  # Ajusta el layout para que no se solapen los elementos
        plt.show()
        
        

        # ----- SEGUNDO GRÁFICO: Eje X con tiempo absoluto (HH:MM:SS) -----        
   
        # Calcular timestamps a partir de stime y dt
        self.tabs = [self.stime + datetime.timedelta(milliseconds=int(i * self.dt * 1000)) 
                      for i in range(self.instant_power_vals_array.shape[1])]
    
        # Sumar 1 hora a los timestamps
        self.tabs = [t + datetime.timedelta(hours=1) for t in self.tabs]
    
        plt.figure(figsize=(12, 6))
        for i in range(self.instant_power_vals_array.shape[0]):  # 33 filas (invertidas)
            plt.plot(self.tabs, self.instant_power_vals_array[i] + i * offset_ip, 
                      label=f'Señal {self.instant_power_vals_array.shape[0] - i}', 
                      linestyle='-', alpha=0.7)

        plt.axvline(x=self.tabs[time_marker], color='red', linestyle='--', label='Coche 1')
        plt.title('33 Señales de Instant Power')
        plt.xlabel('Tiempo (HH:MM:SS)')
        plt.ylabel('Puntos Espaciales')
        plt.yticks(
            np.arange(0, self.instant_power_vals_array.shape[0] * offset_ip, offset_ip), 
            labels=[f'Punto {self.instant_power_vals_array.shape[0] - i}' for i in range(self.instant_power_vals_array.shape[0])]
        )
    
        # Formatear el eje X con HH:MM:SS y mostrar marcas de minuto en minuto
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    
        plt.grid()
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.xticks(rotation=45)  # Rotar etiquetas del eje X para mejor visibilidad
        plt.tight_layout()
        plt.show()

        
        
        
    def calculate_variance_instant_power_plot(self):
            
        """
        Calcula la varianza del instant power en ventanas solapadas y genera gráficos.
        """
        # Parámetros
        fs = 250  # Frecuencia de muestreo
        umbral = 20000  # Umbral máximo para la varianza
    
        # Definir el tamaño de ventana y el overlap en muestras
        window_size = int(window_duration * fs)
        overlap_size = int(overlap_duration * fs)
        step_size = window_size - overlap_size  # Desplazamiento entre ventanas
    
        data_ = self.instant_power_vals_array  # Usar los valores de instant power
    
        # Arrays para almacenar las varianzas de cada fila en ventanas solapadas
        self.varianzas = []
        self.varianzas_umbralizadas = []
    
        # Calcular la varianza por fila en ventanas solapadas
        for row in data_:
            row_variances = []
            row_variances_umbralizadas = []
            # Desplazamiento a lo largo de la señal en pasos de tamaño step_size
            for start in range(0, len(row) - window_size + 1, step_size):
                window = row[start:start + window_size]  # Extrae la ventana
                varianza = np.var(window)  # Calcula la varianza de la ventana
                varianza_umbral = min(varianza, umbral)  # Aplica el umbral
                row_variances.append(varianza)  # Añade la varianza sin umbralizar
                row_variances_umbralizadas.append(varianza_umbral)  # Añade la varianza umbralizada
            self.varianzas.append(row_variances)
            self.varianzas_umbralizadas.append(row_variances_umbralizadas)
    
        # Convertir las listas de varianzas en arrays de numpy para tener matrices uniformes
        self.varianzas = np.array(self.varianzas)
        self.varianzas_umbralizadas = np.array(self.varianzas_umbralizadas)
    
        # Configurar el gráfico
        time_samples = np.arange(self.varianzas.shape[1])  # 281 puntos temporales
    
        # Crear la figura y el eje para los valores umbralizados
        plt.figure(figsize=(12, 8))
    
        # Graficar cada fila de varianzas umbralizadas como líneas continuas
        for i in range(self.varianzas_umbralizadas.shape[0]):  # 33 filas (invertidas)
            # Desplazar la señal en el eje Y
            plt.plot(time_samples, self.varianzas_umbralizadas[i] + i * 2000, label=f'Señal {self.varianzas_umbralizadas.shape[0] - i}', linestyle='-', alpha=0.7)
    
        # Personalizar el gráfico
        plt.title('33 Señales de Varianza del Instant Power (Umbralizadas)')
        plt.xlabel('Muestras de Tiempo')
        plt.ylabel('Puntos Espaciales')
        plt.yticks(
            np.arange(0, self.varianzas_umbralizadas.shape[0] * 2000, 2000), 
            labels=[f'Punto {self.varianzas_umbralizadas.shape[0] - i}' for i in range(self.varianzas_umbralizadas.shape[0])]
        )  # Etiquetas invertidas para los puntos espaciales
    
        plt.grid()
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Mueve la leyenda fuera del gráfico
        plt.tight_layout()  # Ajusta el layout para que no se solapen los elementos
        plt.show()
    
        # Crear la figura y el eje para los valores sin umbralizar
        plt.figure(figsize=(12, 8))
    
        # Graficar cada fila de varianzas sin umbralizar como líneas continuas
        for i in range(self.varianzas.shape[0]):  # 33 filas (invertidas)
            # Desplazar la señal en el eje Y
            plt.plot(time_samples, self.varianzas[i] + i * 2000, label=f'Señal {self.varianzas.shape[0] - i}', linestyle='-', alpha=0.7)
    
        # Personalizar el gráfico
        plt.title('33 Señales de Varianza del Instant Power (Sin Umbralizar)')
        plt.xlabel('Muestras de Tiempo')
        plt.ylabel('Puntos Espaciales')
        plt.yticks(
            np.arange(0, self.varianzas.shape[0] * 2000, 2000), 
            labels=[f'Punto {self.varianzas.shape[0] - i}' for i in range(self.varianzas.shape[0])]
        )  # Etiquetas invertidas para los puntos espaciales
    
        plt.grid()
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Mueve la leyenda fuera del gráfico
        plt.tight_layout()  # Ajusta el layout para que no se solapen los elementos
        plt.show()



    def calculate_peaks_variance_plot(self):
        """
        Calcula la varianza de instant power y genera gráficos con los picos detectados.
        """

        time_samples = np.arange(self.varianzas.shape[1])  # 281 puntos temporales
        
        # Inicializar una lista para almacenar los detalles de los picos de cada fila
        picos_var_ip = []
        
        #Descomentar para ver las 21 gráficas por separado
        
        # ---- Parte 1: Graficar cada fila con sus picos en gráficos separados ---- 
        
        for i, fila in enumerate(self.varianzas):
            # Detectar picos con los parámetros deseados
            peaks, properties = find_peaks(fila, height=height_ip, distance=distance_ip, prominence=prominence_ip)  # Ajusta los parámetros según sea necesario
        
            # Guardar los picos y sus propiedades para cada fila
            picos_var_ip.append({
                'fila o punto espacial': i,
                'picos en el eje x (segun muestras)': peaks,
                'altura o amplitud': properties['peak_heights']
            })
            
            
            # Graficar la fila actual con los picos encontrados en gráficos individuales
            plt.figure(figsize=(10, 3))
            plt.plot(fila, label=f'Punto espacial {self.varianzas.shape[0] - i}')
            plt.plot(peaks, fila[peaks], "x", label='Picos')
            plt.title(f"Picos en el punto espacial {self.varianzas.shape[0] - i}")
            plt.xlabel("Muestras")
            plt.ylabel("Varianza de Instant Power")
            plt.legend()
            plt.show()
        
        
        
        # ---- Parte 2: Graficar todas las señales y sus picos en un solo gráfico ----  SIN UMBRALIZAR
        
        plt.figure(figsize=(12, 8))
        
        for i, fila in enumerate(self.varianzas):
            # Detectar picos en la fila actual
            peaks, properties = find_peaks(fila, height=height_ip, distance=distance_ip, prominence=prominence_ip)  # 1000
        
            # Graficar la señal con desplazamiento y los picos en un solo gráfico
            plt.plot(time_samples, fila + i * offset, label=f'Señal {self.varianzas.shape[0] - i}', linestyle='-', alpha=0.7)
            plt.plot(peaks, fila[peaks] + i * offset, "x")  # Marcar los picos con "x"
            
        
        # Personalizar el gráfico
        plt.title('Delta Strain Variance', fontsize=22)
        plt.xlabel('Samples', fontsize=22)
        plt.ylabel('Spatial Points', fontsize=22)
        plt.yticks(
            np.arange(0, self.varianzas.shape[0] * offset, offset),
            labels=[f'{self.varianzas.shape[0] - i}' for i in range(self.varianzas.shape[0])]
        )  # Etiquetas invertidas para los puntos espaciales
        
        plt.grid()
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Mueve la leyenda fuera del gráfico
        plt.tight_layout()  # Ajusta el layout para evitar solapamientos
        plt.show()
        
        
        # ---- Parte 2: Graficar todas las señales y sus picos en un solo gráfico ----  UMBRALZIADO
        
        plt.figure(figsize=(12, 8))
        
        for i, fila in enumerate(self.varianzas_umbralizadas):
            # Detectar picos en la fila actual
            peaks, properties = find_peaks(fila, height=height_ip, distance=distance_ip, prominence=prominence_ip)  # 1000
        
            # Graficar la señal con desplazamiento y los picos en un solo gráfico
            plt.plot(time_samples, fila + i * offset, label=f'Señal {self.varianzas_umbralizadas.shape[0] - i}', linestyle='-', alpha=0.7)
            plt.plot(peaks, fila[peaks] + i * offset, "x")  # Marcar los picos con "x"
            
        
        # Personalizar el gráfico
        plt.title('Delta Strain Variance', fontsize=22)
        plt.xlabel('Samples', fontsize=22)
        plt.ylabel('Spatial Points', fontsize=22)
        plt.yticks(
            np.arange(0, self.varianzas_umbralizadas.shape[0] * offset, offset),
            labels=[f'{self.varianzas_umbralizadas.shape[0] - i}' for i in range(self.varianzas_umbralizadas.shape[0])]
        )  # Etiquetas invertidas para los puntos espaciales
        
        plt.grid()
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Mueve la leyenda fuera del gráfico
        plt.tight_layout()  # Ajusta el layout para evitar solapamientos
        plt.show()
        
        
        # ---- Invertir y mostrar picos_var_ip ----
        picos_var_ip = picos_var_ip[::-1]
        
        
        
    def seleccionar_indices_picos(self):
        """
        Detecta y visualiza los índices de los picos en las señales de varianzas.

        Args:
            height_ip (float): Altura mínima requerida para los picos.
            distance_ip (int): Distancia mínima entre los picos.
            prominence_ip (float): Prominencia mínima requerida para los picos.

  
        """
  

        plt.figure(figsize=(12, 8))

        # Inicializar una lista para almacenar los detalles de los picos de cada fila
        self.picos_variance_ip = []

        # Itera sobre cada fila (señal) en varianzas y detecta picos
        for i, fila in enumerate(self.varianzas):
            # Detecta los picos en la señal (fila) actual
            peaks_i, properties_i = find_peaks(
                fila, height=height_ip, distance=distance_ip, prominence=prominence_ip
            )

            # Grafica solo los puntos de los picos detectados, desplazados en el eje Y
            plt.scatter(peaks_i, [i] * len(peaks_i), label=f'Señal {self.varianzas.shape[0] - i}', alpha=0.7)

            # Guardar los picos y sus propiedades para cada fila
            self.picos_variance_ip.append({
                'fila o punto espacial': i,
                'picos en el eje x (según muestras)': peaks_i,
                'altura o amplitud': properties_i['peak_heights']
            })

        # Personalizar el gráfico
        plt.title('Visualización de Picos en el Eje X para Varias Señales')
        plt.xlabel('Muestras de Tiempo (Índice de picos)')
        plt.ylabel('Puntos Espaciales (Señales)')
        plt.yticks(
            np.arange(0, self.varianzas.shape[0], 1),
            labels=[f'Punto {self.varianzas.shape[0] - i}' for i in range(self.varianzas.shape[0])]
        )
        plt.grid()
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Mueve la leyenda fuera del gráfico
        plt.tight_layout()
        plt.show()





    def intensidad_circulos_varianza(self, xmin=750, xmax=1500, grid_spacing=25):
        """
        Detecta picos en las señales umbralizadas y genera un gráfico con círculos proporcionales a su amplitud.

        Parámetros:
        - xmin: valor mínimo del eje X.
        - xmax: valor máximo del eje X.
        - grid_spacing: espaciado entre líneas del grid en el eje X.
        """
        plt.figure(figsize=(12, 8))
        colormap = cm.get_cmap('tab20')

        for i, fila in enumerate(self.varianzas_umbralizadas):
            peaks, properties = find_peaks(
                fila, height=height_ip, distance=distance_ip, prominence=prominence_ip
            )
            color = colormap(i / len(self.varianzas_umbralizadas))

            for peak, height in zip(peaks, properties['peak_heights']):
                if xmin <= peak <= xmax:
                    plt.scatter(
                        peak, i * offset, color=color,
                        s=height * 0.025, edgecolors='black', alpha=0.7, marker='o', zorder=5
                    )

        plt.title('Scaled Variance Peaks', fontsize=22)
        plt.xlabel('Samples', fontsize=22)
        plt.ylabel('Spatial Points', fontsize=22)
        plt.yticks(
            np.arange(0, len(self.varianzas_umbralizadas) * offset, offset),
            labels=[f'{len(self.varianzas_umbralizadas) - i}' for i in range(len(self.varianzas_umbralizadas))]
        )

        plt.xlim(xmin, xmax)
        plt.xticks(np.arange(xmin, xmax + 1, grid_spacing))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
                    


    def intensidad_circulos_varianza_plotly(self):
        """
        Detecta picos en las señales umbralizadas y genera un gráfico interactivo 
        con círculos proporcionales a su amplitud (versión Plotly).
        """

        # Crear la figura
        fig = go.Figure()

        # Mapa de colores (similar a 'tab20' en matplotlib)
        colors = plotly.colors.qualitative.T10  # o usa 'Plotly' o 'D3' para más opciones

        # Iterar sobre cada fila (señal)
        for i, fila in enumerate(self.varianzas_umbralizadas):
            # Detectar picos
            peaks, properties = find_peaks(
                fila, height=height_ip, distance=distance_ip, prominence=prominence_ip
            )

            color = colors[i % len(colors)]  # asignar color cíclico

            # Agregar los picos como scatter
            fig.add_trace(go.Scatter(
                x=peaks,
                y=[i * offset] * len(peaks),
                mode='markers',
                marker=dict(
                    size=properties['peak_heights'] * 0.025,
                    color=color,
                    line=dict(color='black', width=1),
                    opacity=0.7,
                    symbol='circle'
                ),
                showlegend=False
            ))

        # Personalizar ejes
        fig.update_layout(
            title='Scaled Variance Peaks1',
            xaxis_title='Samples',
            yaxis_title='Spatial Points',
            yaxis=dict(
                tickmode='array',
                tickvals=np.arange(0, len(self.varianzas_umbralizadas) * offset, offset),
                ticktext=[f'{len(self.varianzas_umbralizadas) - i}' for i in range(len(self.varianzas_umbralizadas))]
            ),
            font=dict(size=22),
            template='simple_white',
            height=800,
            width=1200
        )

        fig.show()
    
        
        
        
    
    
    def creaccion_matrices_binarias(self):
        """
        Crea una matriz binaria a partir de los picos detectados, la visualiza,
        
        La matriz final modificada se guarda en self.new_matrix
        """
        # Lista para almacenar los índices de los picos de todas las señales
        peaks_indices1 = []
        all_signals_binary = []
    
        # Crear matriz binaria inicial
        for i, fila in enumerate(self.varianzas): # usamos varianzas SIN UMBRAL
            peaks_i, properties_i = find_peaks(fila, height=height_ip, distance=distance_ip, prominence=prominence_ip)
            peaks_indices1.append(peaks_i)
            
            # Crear una señal binaria de ceros, donde los picos serán 1
            signal_binary = np.zeros_like(fila)
            signal_binary[peaks_i] = 1
            all_signals_binary.append(signal_binary)
    
        # Convertir la lista de señales binarias a una matriz
        self.binary_matrix = np.array(all_signals_binary)
    
        # Visualizar matriz binaria original
        rows, cols = np.where(self.binary_matrix == 1)
        rows = self.binary_matrix.shape[0] - 1 - rows
    
        plt.figure(figsize=(12, 8))
        plt.scatter(cols, rows, color='blue', label='1s en la matriz', s=10)
        plt.title('Visualización de 1s en la Matriz Binaria')
        plt.xlabel('Muestras (Eje X)')
        plt.ylabel('Filas (Eje Y)')
        plt.yticks(
            np.arange(0, self.binary_matrix.shape[0]),
            labels=[f'Fila {i}' for i in range(self.binary_matrix.shape[0])]
        )
        plt.gca().invert_yaxis()
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        # Crear una copia de la matriz para modificar
        modified_matrix = self.binary_matrix.copy()
        
        # Funciones internas para el procesamiento
        def check_expanding_range(row, col, original_matrix):
            search_rows = min(1, row)
            base_range = 20
            
            for i in range(1, search_rows + 1):
                check_row = row - i
                if check_row < 0:
                    break
                    
                current_range = base_range * i
                search_start = max(0, col - current_range)
                search_end = min(col + current_range + 1, original_matrix.shape[1])
                
                if 1 in original_matrix[check_row, search_start:search_end]:
                    return True
            return False
    
        def calculate_trend_position(row, col):
            prev_points = []
            search_rows = min(2, row)
            
            for i in range(search_rows):
                prev_row = row - (i + 1)
                if prev_row >= 0:
                    search_start = max(0, col - 40)
                    search_end = min(col + 40, modified_matrix.shape[1])
                    prev_row_ones = np.where(modified_matrix[prev_row, search_start:search_end] == 1)[0]
                    
                    if len(prev_row_ones) > 0:
                        actual_position = prev_row_ones[0] + search_start
                        prev_points.append((prev_row, actual_position))
            
            if len(prev_points) >= 2:
                diffs = []
                for i in range(1, len(prev_points)):
                    dx = prev_points[i][1] - prev_points[i-1][1]
                    dy = prev_points[i][0] - prev_points[i-1][0]
                    if dy != 0:
                        diffs.append(dx/dy)
                
                if diffs:
                    avg_diff = sum(diffs) / len(diffs)
                    predicted_pos = int(col + avg_diff)
                    return min(max(0, predicted_pos), modified_matrix.shape[1] - 1)
            return min(col + 15, modified_matrix.shape[1] - 1)
        
        def propagate_ones(row, col):
            if row >= modified_matrix.shape[0] - 1:
                return
                
            if not check_expanding_range(row, col, self.binary_matrix):
                return
                
            search_start = max(0, col - 25)
            search_end = min(col + 25 + 1, modified_matrix.shape[1])
            
            found_one = False
            search_range = modified_matrix[row + 1, search_start:search_end]
            
            if 1 in search_range:
                found_one = True
            
            if not found_one:
                new_one_position = calculate_trend_position(row, col)
                modified_matrix[row + 1, new_one_position] = 1
                propagate_ones(row + 1, new_one_position)
        
        # Aplicar la propagación de unos
        for row in range(self.binary_matrix.shape[0] - 1):
            current_ones = np.where(modified_matrix[row] == 1)[0]
            for col in current_ones:
                propagate_ones(row, col)
        
        # Guardar la matriz modificada como atributo de la clase
        self.new_matrix = modified_matrix
        
        # Visualizar la matriz modificada
        rows, cols = np.where(self.new_matrix == 1)
        rows = self.new_matrix.shape[0] - 1 - rows
    
        plt.figure(figsize=(12, 8))
        plt.scatter(cols, rows, color='blue', label='1s en la matriz', s=10)
        plt.title('Visualización de 1s en la Matriz Binaria (Con 1s inventados)')
        plt.xlabel('Muestras (Eje X)')
        plt.ylabel('Filas (Eje Y)')
        plt.yticks(
            np.arange(0, self.new_matrix.shape[0]),
            labels=[f'Fila {i}' for i in range(self.new_matrix.shape[0])]
        )
        plt.gca().invert_yaxis()
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
            

    def trayectorias_finales(self, rango_derecha=24, rango_izquierda=4, max_filas_superiores=5):
        """
        Conecta picos detectados en una matriz binaria (0 y 1) mediante tres pasadas.
    
        Params:
        - new_matrix: numpy array 2D donde los valores 1 representan los picos.
        - rango_derecha: Rango para conexiones hacia la derecha en filas contiguas.
        - rango_izquierda: Rango para conexiones hacia la izquierda en filas contiguas.
        - max_filas_superiores: Máximo número de filas superiores a considerar en la tercera vuelta.
        """
        # Identificar los índices de los picos (valores 1 en la matriz)
        peaks_all_rows_i = [np.where(row == 1)[0] for row in self.new_matrix]
    
        # Crear registros de conexiones
        lineas_salientes = [set() for _ in range(len(peaks_all_rows_i))]
        lineas_entrantes = [set() for _ in range(len(peaks_all_rows_i))]
    
        plt.figure(figsize=(12, 8))
    
        # Graficar picos originales
        for i, peaks_i in enumerate(peaks_all_rows_i):
            plt.scatter(peaks_i, [i] * len(peaks_i), label=f'Señal {i + 1}', alpha=0.7)
    
        # Colores de líneas
        line_color_original = 'k'  # Negro para conexiones originales
        line_color_segunda = 'r'   # Rojo para segunda vuelta
        line_color_tercera = 'orange'   # Morado para la tercera vuelta
    
        # Primera vuelta: Conectar picos de izquierda a derecha con forma diagonal "/"
        for i in range(len(peaks_all_rows_i) - 1):
            current_peaks = peaks_all_rows_i[i]  # Picos en la fila actual
            next_peaks = peaks_all_rows_i[i + 1]  # Picos en la siguiente fila
    
            for peak in current_peaks:
                for next_peak in next_peaks:
                    # Conectar solo si el siguiente pico es mayor y está dentro del rango
                    if 0 < (next_peak - peak) <= rango_derecha:
                        plt.plot([peak, next_peak], [i, i + 1], color=line_color_original, linestyle="-", linewidth=1.5)
                        # Registrar conexiones originales
                        lineas_salientes[i].add(peak)
                        lineas_entrantes[i + 1].add(next_peak)
                        break  # Salir después de conectar el primer pico válido
    
        # Segunda vuelta: Buscar en filas contiguas en el rango izquierdo
        for i in range(len(peaks_all_rows_i) - 1):
            current_peaks = peaks_all_rows_i[i]  # Picos en la fila actual
            next_peaks = peaks_all_rows_i[i + 1]  # Picos en la siguiente fila
    
            for peak in current_peaks:
                if peak in lineas_salientes[i]:  # Si ya tiene una línea saliente, saltamos
                    continue
    
                for next_peak in next_peaks:
                    # Conectar solo si el siguiente pico es menor y está dentro del rango izquierdo
                    if 0 <= (peak - next_peak) <= rango_izquierda:
                        plt.plot([peak, next_peak], [i, i + 1], color=line_color_tercera, linestyle='-', linewidth=1.5)
                        # Registrar conexiones de la tercera vuelta
                        lineas_salientes[i].add(peak)
                        lineas_entrantes[i + 1].add(next_peak)
                        break  # Salir después de conectar el primer pico válido
    
        # Tercera vuelta: Buscar conexiones en filas superiores progresivamente
        for i in range(len(peaks_all_rows_i)):  # Iterar desde abajo hacia arriba
            current_peaks = peaks_all_rows_i[i]
    
            for peak in current_peaks:
                if peak in lineas_salientes[i]:  # Si ya tiene línea saliente hacia arriba, saltamos
                    continue
    
                # Búsqueda progresiva en filas superiores
                fila_actual = i
                conectado = False
    
                for offset in range(2, min(max_filas_superiores + 1, len(peaks_all_rows_i) - fila_actual)):
                    next_row_index = fila_actual + offset  # Índice de la fila superior
                    next_peaks = peaks_all_rows_i[next_row_index]
    
                    # Calcular rango dinámico basado en `rango_derecha`
                    rango_actual = rango_derecha * offset
    
                    # Buscar el primer pico válido dentro del rango actual
                    for next_peak in next_peaks:  # Cambio: iterar de izquierda a derecha
                        if (next_peak > peak and  # Subir en diagonal (de izquierda a derecha)
                            next_peak not in lineas_entrantes[next_row_index] and
                            0 < (next_peak - peak) <= rango_actual):
    
                            # Dibujar conexión y registrar
                            plt.plot([peak, next_peak],
                                     [fila_actual, next_row_index],
                                     color=line_color_segunda, linestyle='--', linewidth=1.5)
                            lineas_salientes[fila_actual].add(peak)
                            lineas_entrantes[next_row_index].add(next_peak)
                            conectado = True
                            break
    
                    if conectado:
                        break  # Si se conectó, no seguir buscando en filas más arriba
                        
                        
    
        # Personalizar el gráfico
        plt.title('Trajectories', fontsize=22)  # Título con tamaño grande
        plt.xlabel('Samples', fontsize=22)  # Eje X con tamaño grande
        plt.ylabel('Spatial Points', fontsize=22)  # Eje Y con tamaño grande
        
        # Personalizar las etiquetas de los ejes con mayor tamaño
        plt.xticks(fontsize=18)  # Aumentar tamaño de las etiquetas del eje X
        plt.yticks(
            np.arange(0, self.new_matrix.shape[0], 1),
            labels=[f'{self.new_matrix.shape[0] - i}' for i in range(self.new_matrix.shape[0])],
            fontsize=18  # Aumentar tamaño de las etiquetas del eje Y
        )
        
        # Ajustar el layout y mostrar
        plt.tight_layout()
        plt.show()
        
            



    def calculo_SNR_punto_espacial(self, H, PS, channel_offset=100):
        """
        Calcula el SNR de una señal de un canal seleccionado interactivamente.
    
        Retorna:
        - SNR_total : float, SNR calculado en dB.
        """
        H.cutX(1000, 3520)

    
        # Seleccionar la señal del punto espacial
        signal = H.da[PS, :]
    
        # Variables para almacenar las regiones seleccionadas
        noise_region = []
        signal_region = []
        selection_complete = False  # Bandera para determinar si la selección está completa
    
        # Función de selección de regiones
        def onselect(eclick, erelease):
            """
            Función de callback para capturar las regiones seleccionadas.
            """
            nonlocal noise_region, signal_region, selection_complete
    
            x1, x2 = eclick.xdata, erelease.xdata
            region = [int(min(x1, x2)), int(max(x1, x2))]
            print(f"Región seleccionada: {region}")
    
            if not noise_region:
                noise_region = region
                print(f"Ventana de ruido seleccionada: {noise_region}")
            elif not signal_region:
                signal_region = region
                print(f"Ventana de señal seleccionada: {signal_region}")
                selection_complete = True  # Ambas regiones han sido seleccionadas
                plt.close()  # Cierra la gráfica una vez que ambas regiones están seleccionadas
    
        # Graficar la señal para selección interactiva
        print("Selecciona las regiones de ruido y señal.")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(signal, label=f"Punto espacial {PS + channel_offset}")
        ax.set_title("Selecciona las regiones de ruido y señal con el ratón")
        ax.set_xlabel("Muestras")
        ax.set_ylabel("Amplitud")
        ax.legend()
        ax.grid()
    
        # Herramienta de selección de rectángulos
        rectangle_selector = RectangleSelector(
            ax,
            onselect,
            interactive=True
        )
    
        # Usar plt.pause() para permitir la interacción mientras se espera la selección
        while not selection_complete:
            plt.pause(0.1)  # Permitir que la gráfica se actualice y espere interacción
    
        # Validar si ambas regiones están definidas
        if len(noise_region) == 0 or len(signal_region) == 0:
            print("Error: No se seleccionaron correctamente las regiones de ruido o señal.")
            print("Vuelve a ejecutar el script y selecciona ambas regiones.")
            return None
    
        # Asegurarse de que las regiones estén dentro del rango válido
        noise_region[0] = max(0, noise_region[0])
        noise_region[1] = min(H.da.shape[1], noise_region[1])
        signal_region[0] = max(0, signal_region[0])
        signal_region[1] = min(H.da.shape[1], signal_region[1])
    
        # print(f"Región de ruido seleccionada ajustada: {noise_region}")
        # print(f"Región de señal seleccionada ajustada: {signal_region}")
    
        # # Calcular la energía para el canal seleccionado
        # print("Calculando SNR total para el canal seleccionado...")
    
        # Obtener las ventanas de ruido y señal para el canal
        w_noise = signal[noise_region[0]:noise_region[1]]
        w_signal = signal[signal_region[0]:signal_region[1]]
    
        # Calcular la energía promedio para ruido y señal
        energy_noise = np.sum(w_noise**2) / len(w_noise)
        energy_signal = np.sum(w_signal**2) / len(w_signal)
    
        # Calcular SNR total en dB
        SNR_total = 10 * np.log10(energy_signal / energy_noise)
    
        print(f"SNR Total para el Punto Espacial {PS + channel_offset}: {SNR_total:.2f} dB")
    
        # Graficar la señal con las regiones seleccionadas resaltadas
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(signal, label=f"Punto Espacial {PS + channel_offset}")
        ax.axvspan(noise_region[0], noise_region[1], color='red', alpha=0.3, label="Ruido")
        ax.axvspan(signal_region[0], signal_region[1], color='green', alpha=0.3, label="Señal")
        ax.set_title(f"SNR Total para el Punto Espacial {PS + channel_offset}: {SNR_total:.2f} dB")
        ax.set_xlabel("Muestras")
        ax.set_ylabel("Amplitud")
        ax.legend()
        ax.grid()
        plt.show()
    

        


        

    def ejemplo_conteo_vehiculos_SNR(self):

            
        SNR_Santiago = {
        
            "SNR_1" : [2.90, 6.35, 2.18],
            
            "SNR_2" : [1.99, 13.09, 1.04],
            
            "SNR_3" : [7.01, 10.68, 0.95],
            
            "SNR_4" : [5.33, 6.90, 4.36],
            
            "SNR_5" : [5.86, 7.76, 3.93],
            
            "SNR_6" : [8.18, 15.08, 6.44],
            
            "SNR_7" : [1.14, 5.52, 1.00],
            
            "SNR_8" : [7.37, 10.06, 4.79],
            
            "SNR_9" : [7.49, 21.82, 7.53],
            
            "SNR_10" : [3.03, 8.56, 1.82],
            
            "SNR_11" : [1.41, 11.23, 3.03],
            
            "SNR_12" : [3.25, 9.03, 3.95],
            
            "SNR_13" : [4,40, 9.82, 3.21],
            
            "SNR_14" : [5.64, 10.67, 2.66],
            
            "SNR_15" : [6.98, 21.17, 12.56],
            
            "SNR_16" : [14.76, 24.51, 17.85],
            
            "SNR_17" : [19.32, 16.79, 10.88],
            
            "SNR_18" : [9.75, 18.03, 9.47],
            
            "SNR_19" : [10.13, 21.76, 10.43],
            
            "SNR_20" : [16.34, 24.38, 11.73],
            
            "SNR_21" : [12.37, 21.57, 13.92],
        
        }
            
        # Calcular las medias de SNR y crear un nuevo diccionario
        self.SNR_Santiago_media = {}
        for clave, valores in SNR_Santiago.items():
            media = np.mean(valores)
            self.SNR_Santiago_media[clave] = round(media, 2)
    
        # # Imprimir las medias calculadas
        # print("Medias de SNR calculadas:")
        # for clave, media in  self.SNR_Santiago_media.items():
        #     print(f"{clave}: {media:.2f}")
    
        # Almacenar los picos en cada fila
        picos_conteo = []
    
        for i, fila in enumerate(self.varianzas):
            peaks_i, _ = find_peaks(fila, height=height_ip, distance=distance_ip, prominence=prominence_ip)
            picos_conteo.append(peaks_i)
    
        # Invertir la lista de picos para alinearse con las filas
        picos_conteo = picos_conteo[::-1]
    
        # Convertir los valores de SNR a una lista ordenada por las claves
        snr_lista = [ self.SNR_Santiago_media[f"SNR_{i+1}"] for i in range(len(picos_conteo))]
    
        # Filtro: Usar solo las filas 3, 6, 17, 18 (ajustando a índices base 0)
        selected_rows = [2, 5, 16, 17]
        selected_weights = [snr_lista[i] for i in selected_rows]
    
        # Normalizar los pesos para que sumen 1
        normalized_weights = np.array(selected_weights) / sum(selected_weights)
    
        # Inicializar el conteo ponderado de vehículos
        weighted_vehicle_count = 0
    
        for i, row_index in enumerate(selected_rows):
            # Tamaño de la fila en picos_conteo corresponde al número de vehículos detectados
            num_vehicles_in_row = len(picos_conteo[row_index])
    
            # Ponderar por el peso normalizado
            weighted_vehicle_count += num_vehicles_in_row * normalized_weights[i]
    
        # Redondear hacia arriba para obtener el conteo final
        self.final_vehicle_count = int(np.ceil(weighted_vehicle_count))
    
        # Calcular la media de los vehículos detectados por fila seleccionada
        vehicle_counts_per_row = [len(picos_conteo[row]) for row in selected_rows]
        mean_vehicle_count = int(np.round(np.mean(vehicle_counts_per_row)))
    
        # Graficar el resultado
        plt.figure(figsize=(12, 6))
        plt.bar(
            [f"Fila {i+1}" for i in selected_rows],
            vehicle_counts_per_row,
            label="Vehículos por fila (antes de ponderar)",
            alpha=0.7
        )
    
        # Añadir línea roja discontinua para la media real
        plt.axhline(y=mean_vehicle_count, color='red', linestyle='--', label=f'Media: {mean_vehicle_count +1} vehículos')
        plt.title("Conteo de Vehículos Basado en Picos y SNR")
        plt.xlabel("Filas seleccionadas")
        plt.ylabel("Número de vehículos detectados")
        plt.grid(axis="y")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        # Imprimir resultados
        print("\nResultados finales:")
        print(f"Número total de vehículos detectados (ponderado): {self.final_vehicle_count}")
        print(f"Filas seleccionadas: {[i+1 for i in selected_rows]}")
        print(f"Pesos normalizados asignados: {normalized_weights}")
    
    
    
        
        # Preparar los datos para el gráfico
        x = list(SNR_Santiago.keys())  # Eje x: nombres de SNR
        y1 = [valores[0] for valores in SNR_Santiago.values()] # Primera medicion
        y2 = [valores[1] for valores in SNR_Santiago.values()] # Segunda medicion
        y3 = [valores[2] for valores in SNR_Santiago.values()] # Tercera medicion
        
        # Crear el gráfico
        plt.figure(figsize=(15, 6)) #Ajusta el tamaño del gráfico para mejor visualización.
        plt.plot(x, y1, label="Medición 1", marker='o')
        plt.plot(x, y2, label="Medición 2", marker='x')
        plt.plot(x, y3, label="Medición 3", marker='^')
        
        # Añadir etiquetas y título
        plt.xlabel("Puntos SNR")
        plt.ylabel("Valor SNR")
        plt.title("Valores SNR para cada punto espacial")
        plt.xticks(rotation=45, ha='right') # Rotar las etiquetas del eje x para mejor legibilidad
        plt.legend()
        plt.grid(True)
        
        # Mostrar el gráfico
        plt.tight_layout() #Ajusta los parámetros del subplot para proporcionar un relleno automático.
        plt.show()
        
        
            
        

    # # AMPA

    # def prctile(self, x, p):
    #     if isinstance(p, collections.abc.Iterable):
    #         return np.percentile(x, p)
    #     return np.percentile(x, [p])[0]
    
    
    
    
    # def ampa(self, x, fs, threshold=None, L=None, L_coef=3.,
    #          noise_thr=90, bandwidth=3., overlap=1., f_start=0.7, max_f_end=15.,
    #          U=12., peak_window=1.):
    #     """ Implementación de AMPA. """
    #     if L is None:
    #         L = [30., 20., 10., 5., 2.5]
    
    #     x = x - np.mean(x)  # Eliminamos la media
    #     fs = float(fs)
    #     peak_window = round(peak_window * fs / 2.)
    #     step = bandwidth - overlap
    
    #     flo = np.arange(f_start, min(fs / 2. - bandwidth, max_f_end), step)
    #     fhi = flo + bandwidth
    #     z = np.zeros((len(flo), len(x)))
    
    #     for i in range(len(flo)):
    #         h_aux = 8 - (np.arange(32) / 4.)
    #         h0 = np.zeros(512)
    #         h0[0:32] = h_aux * np.cos(2. * np.pi * ((flo[i] + fhi[i]) / 2.) *
    #                                   np.arange(32) / fs)
    #         h0o = np.imag(hilbert(h0))
    #         xa = fftconvolve(x, h0)[:len(x)]
    #         xao = fftconvolve(x, h0o)[:len(x)]
    #         y0 = np.sqrt((xa ** 2) + (xao ** 2))
    #         thr = self.prctile(y0, noise_thr)
    #         z[i, :] = (y0 / thr) * (y0 > thr) + (y0 <= thr)
    
    #     ztot = np.sum(z, 0)
    #     lztot = np.log10(ztot) - np.min(np.log10(ztot)) + 1e-2
    #     Ztot = np.zeros((len(L), len(x)))
    
    #     for i in range(len(L)):
    #         l = int(L[i] * fs)
    #         B = np.zeros(2 * l)
    #         B[0:l] = range(1, l + 1)
    #         B[l:2 * l] = L_coef * (np.arange(1, l + 1) - (l + 1))
    #         B = B / np.sum(np.abs(B))
    #         Zt = fftconvolve(lztot, B)[:len(x)]
    #         Zt = Zt * (Zt > 0)
    #         Ztot[i, :-l] = np.roll(Zt, -l)[:-l]
    
    #     ZTOT = np.prod(Ztot, 0)[:-(int(max(L) * fs))]
    #     ZTOT = U + np.log10(np.abs(ZTOT) + (10 ** -U))
    #     #event_t = find_peaks(ZTOT, threshold, distance=peak_window)[0]
    #     event_t = findpeaks.find_peaks(ZTOT, threshold, order=peak_window * fs)
    #     return event_t, ZTOT
    






      
 

      
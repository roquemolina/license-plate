**1\. Introducción**
====================

Este documento describe el desarrollo de un sistema ALPR **(Reconocimiento Automático de Matrículas)** que utiliza YOLO para la detección de vehículos y matrículas y PaddleOCR para el reconocimiento de texto.

Para comprender completamente el alcance y los matices del proyecto, describiré algunos de los factores clave que impulsaron su concepción:

1\. **Objetivo principal**: El objetivo principal es desarrollar un sistema ALPR.

2\. **Funcionalidad y portabilidad sin conexión**: El dispositivo debe funcionar en entornos sin internet por cable ni alimentación de corriente alterna (CA). Debe poder instalarse en exteriores a nivel de calle.

3\. **Procesamiento en el dispositivo**: El sistema analizará los datos localmente para detectar las matrículas. Al identificar una coincidencia, transmitirá únicamente la información esencial, como el número de matrícula detectado, el fotograma de la imagen correspondiente y la marca de tiempo. Este enfoque minimiza los costos operativos, ya que la transmisión de secuencias de video completas es prohibitivamente cara en comparación con la transmisión de datos concisos y relevantes.

**Descripción general del proceso**
-----------------------------------

Para lograr nuestro objetivo principal de reconocimiento automático de matrículas (ALPR), el flujo de trabajo consta de dos etapas clave:

### **Etapa 1: Entrenamiento del modelo de IA**

- **Objetivo**: Preparar un modelo para detectar e interpretar matrículas con precisión.

- **Enfoque**:

- **Opción 1**: Entrenar un modelo personalizado desde cero utilizando datos propios y configuraciones personalizadas.

- **Opción 2**: Ajustar un modelo preentrenado (aprendizaje por transferencia) para adaptarlo a nuestro caso de uso específico. Esto equilibra la eficiencia y la precisión al aprovechar las arquitecturas existentes.

### **Etapa 2: Detección y Reconocimiento**

- **Flujo de trabajo**:

1\. **Detectar vehículos**: Identificar vehículos (automóviles, camiones, etc.) en el fotograma utilizando un modelo **YOLO** preentrenado (entrenado con el conjunto de datos COCO).

2\. **Localizar matrículas**: Extraer regiones de interés (ROI) que contienen las matrículas de los vehículos detectados.

3\. **Leer matrículas**: Utilizar PaddleOCR, una herramienta de reconocimiento óptico de caracteres (OCR) optimizada para texto de baja resolución o distorsionado, para decodificar los caracteres alfanuméricos de las matrículas.

2\. Instalación
=================

**Instalar Python**
------------------

- ​​**Propósito**: Python es el lenguaje de programación necesario para ejecutar la aplicación. **Recomendaciones**:

El proyecto se ha probado sin problemas en arquitecturas de CPU Intel, AMD y ARM, así como en NVIDIA y GPU integradas.

**Evite Python 3.13**: Esta versión es demasiado reciente y puede presentar problemas de compatibilidad. Utilice Python 3.11 en su lugar, ya que es estable y totalmente compatible con las dependencias del proyecto.

Siga las guías de instalación específicas de su sistema operativo (Windows, macOS o Linux).

**Configurar un entorno virtual**
--------------------------------

**Propósito**: Los entornos virtuales aíslan las dependencias del proyecto para evitar conflictos con la instalación de Python de todo el sistema.

**Herramientas**:

**venv** (módulo integrado de Python): Ligero y suficiente para la mayoría de los casos de uso.

**Conda** (vía Anaconda/Miniconda): Ideal para gestionar dependencias complejas y compatibilidad multiplataforma. - **Instrucciones**:

- El proyecto se desarrolló con **venv**, pero Conda es igualmente viable. Consulta la documentación oficial de [venv](https://docs.python.org/3/library/venv.html) o [Conda](https://docs.conda.io/) según tus preferencias.

- **Recordatorio importante**: Instala siempre las dependencias (mediante **`pip`**) **dentro del entorno virtual activado**.

**Instalar dependencias de código**
-----------------------------

Este paso instala las bibliotecas principales necesarias para la canalización implementada. El proceso de instalación **varía ligeramente según se utilice una GPU o una CPU** para el procesamiento de datos.

### **Instalar Node.js**

Node.js se utiliza para ejecutar un servidor backend ligero que gestiona los datos entrantes de las matrículas mediante solicitudes HTTP y los guarda en un archivo CSV. Posteriormente, podremos gestionar datos o recibir tramas.

- **Instrucciones específicas del sistema operativo**:

- **Windows/MacOS**: Descarga la versión LTS del [Sitio oficial de Node.js](https://nodejs.org/).

- **Linux**: Usa tu gestor de paquetes (p. ej., **`sudo apt install nodejs npm`** para Ubuntu/Debian).

Instala las dependencias de YOLO y paddleOCR
---------------------------------------

### **Creación de entornos virtuales**

```

python -m venv /path/to/new/virtual/environment #Si usas Linux, usa python3

```

### Ejecutando venv

| Platform | Shell | Command to activate virtual environment |
| --- | --- | --- |
| POSIX | bash/zsh | `$ source *<venv>*/bin/activate` |
|  | fish | `$ source *<venv>*/bin/activate.fish` |
|  | csh/tcsh | `$ source *<venv>*/bin/activate.csh` |
|  | pwsh | `$ *<venv>*/bin/Activate.ps1` |
| Windows | cmd.exe | `C:\\> *<venv>*\\Scripts\\activate.bat` |
|  | PowerShell | `PS C:\\> *<venv>*\\Scripts\\Activate.ps1` |

Al activar un entorno virtual de Python (creado con **`venv`** o **`conda`**), la interfaz de línea de comandos (CLI) mostrará el nombre del entorno entre paréntesis al principio del mensaje.

```

(your-venv-name) user@device ~/$

```

### Instalar dependencias

**opencv** <https://pypi.org/project/opencv-python/#frequently-asked-questions> En Python, `import cv2` es el comando que se utiliza para importar la biblioteca OpenCV. Una vez instalado, `cv2` da acceso a todas las funciones y clases que OpenCV ofrece para el procesamiento de imágenes, la visión artificial y el aprendizaje automático. En nuestro caso, lo usamos para capturar cada fotograma.

```

pip install opencv-python-headless

```

**request** (biblioteca HTTP sencilla pero elegante) <https://pypi.org/project/requests/>

```

pip install requests

```

**pytorch** <https://pytorch.org/get-started/locally/>

PyTorch es una biblioteca fundamental para nuestro proyecto. Tanto **Ultralytics** (modelos YOLO) como **PaddleOCR** ​​se basan en PyTorch para procesar **tensores**, matrices numéricas multidimensionales que sirven como la estructura de datos principal para los cálculos de aprendizaje automático. Siga el enlace para instalarlo. **Consideraciones de instalación**

**Requisitos del sistema**:

- **Sistema operativo**: Compatible con Windows, Linux y macOS.

- **Lenguaje**: Python (se recomienda la versión 3.8 o superior).

- **Hardware**: CPU o GPU NVIDIA (compatible con CUDA).

**GPU vs. CPU**:

- **Versión de CPU**: Usar si no hay una GPU NVIDIA disponible. Es más lenta, pero universalmente compatible.

- **Versión de GPU**: Requiere **CUDA 11.8**, que garantiza la compatibilidad con Ultralytics y PaddleOCR.

**ultralytics** <https://github.com/ultralytics/ultralytics?tab=readme-ov-file>

**Ultralytics** permite ejecutar y entrenar modelos YOLO. En este proyecto implementado, lo usamos para cargar:

1\. **YOLOv8 preentrenados** (del conjunto de datos COCO) para la detección de vehículos (**`yolo/models/yolo11s.pt`**). 2. **Detector de matrículas personalizado** (**`yolo/models/license_plate_small_v1.pt`**), preentrenado por nosotros (imágenes 4k, 100 épocas).

**Notas clave**:

- Ambos modelos se utilizan directamente; no se requiere entrenamiento adicional para la implementación.

- **Posibles mejoras**:

- Entrene un modelo de matrículas más grande (más épocas + conjunto de datos ampliado).

- Experimente con variantes de YOLO más grandes (p. ej., YOLOv8m/l) para mejorar la precisión con una ligera reducción de velocidad.

```

pip install ultralytics

```

paddleOCR <https://paddlepaddle.github.io/PaddleOCR/latest/en/ppocr/quick_start.html>

El proceso de instalación de PaddleOCR depende de si desea utilizar la aceleración de la GPU o el procesamiento solo de la CPU. Sigue estos pasos:

**Instalar PaddlePaddle (Marco base)**

```

pip install paddlepaddle-gpu # Para usuarios de GPU (NVIDIA con CUDA 11.x)

o

pip install paddlepaddle # Para usuarios de CPU

```

**Instalar PaddleOCR**

```

pip install "paddleocr>=2.0.1" # Se recomienda usar la versión 2.0.1 o superior

```

Tras completar la instalación, sigue estos pasos para iniciar el sistema:

### **Configuración**

**Dirección del servidor**:

- Actualiza el punto final de la API en tu script de Python (variable **`API_URL`**) para que coincida con la IP/dominio de tu servidor.

- **Para pruebas remotas**: Asegúrate de que las reglas del firewall permitan el tráfico entrante/saliente en el puerto designado (p. ej., el puerto 3000).

**Fuente de entrada**:

- Configurar **`INPUT_SOURCE`** en el script de Python para usar:

- Archivo de vídeo local (p. ej., **`./input.mp4`**)

- Transmisión de la cámara (usar **`0`** para la cámara web predeterminada)

### **Iniciar el servidor Node.js**

```

npm install # Instalar las dependencias si aún no lo han hecho

node app.js

```

### **Ejecutar la canalización ALPR**

```

# Activar primero el entorno virtual (si se usa)

source your-venv/bin/activate # Linux/macOS

your-venv\\Scripts\\activate # Windows

python3.11 license_reader_v5.py # Usar el nombre de archivo del script

```

3\. Get Data to TRAIN:

----------------------

One critical facet of the project involves training custom models. Below is the workflow and key requirements:

**Dataset Preparation**

-   **Option 1: Create our Own Dataset**

    -   **Tools**:

        -   **[CVAT.ai](http://CVAT.ai)** (Computer Vision Annotation Tool): Web-based platform for collaborative labeling.

        -   **LabelImg**: Lightweight desktop tool for bounding box annotation.

    -   **Process**:

        -   Annotate images with bounding boxes (license plates, vehicles).

        -   Generate annotation files (**`.txt`** per image) in YOLO format

```

<class_id> <x_center> <y_center> <width> <height>

```

    -   **Note**: Manual annotation is time-intensive. Initial tests used only ~4k images.

-   **Option 2: Use Existing Datasets**

    -   **Sources**:

        -   **Open Images V7**: Large-scale dataset with diverse vehicle classes.

        -   **RoboFlow**: Preprocessed datasets with YOLO-compatible formats.

        -   **Kaggle**: Community-shared datasets (search for "license plates").

### **Training Workflow**

-   **Format Conversion**: Ensure dataset annotations match YOLO requirements.

-   **Configure YOLO**:

    -   Update **`data.yaml`** to specify class names and dataset paths.

    -   Adjust hyperparameters (epochs, batch size) in **`hyp.yaml`**.

1.  **Run Training**:

```
yolo detect train data=config.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0
```

**Key Recommendations**

-   **Dataset Size**: Aim for ≥10k annotated images for robust license plate detection.

-   **Model Variants**: Experiment with larger YOLO architectures (e.g., **`yolov8m.pt`**) for improved accuracy.

-   **Validation**: Split data into 80% train, 15% validation, 5% test sets.

**Interesting links:**

<https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F0pcr&id=4e7c2085ed1a30f9>

To download those images we can use this library, but there is other ones <https://github.com/theAIGuysCode/OIDv4_ToolKit>.

<https://github.com/pjreddie/darknet/blob/master/data/coco.names> COCO NAMES to use library
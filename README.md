# Trabajo Practico N°3 - Procesamiento de Imagenes
# Detector de tirada de dados

## Objetivo
El objetivo de este trabajo es desarrollar un detector de tirada de dados utilizando técnicas de procesamiento de imagenes.
En particular, desarrollar un algoritmo para detectar automaticamente cuando se detienen los
dados y leer el numero obtenido en cada uno.

## Pasos de configuración

### 1) Clonar este repositorio

Para comenzar, clona este repositorio en tu máquina local. Ejecuta el siguiente comando en tu consola, asegurándote de estar en la carpeta donde deseas guardar el proyecto:

`` git clone https://github.com/NancyCima/Detector-de-Tirada-de-Dados.git``

### 2) Instalar Python
Descarga e instala Python desde el siguiente enlace:
- [Python Downloads](https://www.python.org/downloads/)

Asegúrate de agregar Python al PATH durante la instalación.

### 3) Instalar algún IDE para edición/debug
Puedes utilizar alguno de los siguientes IDEs recomendados para trabajar en este proyecto:

- [Visual Studio Code](https://code.visualstudio.com/)
- [PyCharm](https://www.jetbrains.com/es-es/pycharm/)
- También puedes utilizar cualquier otro editor de tu preferencia.

### 4) Crear entorno virtual
Para crear un entorno virtual en el proyecto:

- Crear el entorno virtual:
  ```bash
  python -m venv “directorio”
  ```

- Activar el entorno virtual (Windows):
  ```bash
  directorio\Scripts\activate.bat
  ```

- En sistemas Linux/MacOS, la activación se realiza con:
  ```bash
  source directorio/bin/activate
  ```

### 5) Instalar paquetes necesarios
Instala los paquetes necesarios utilizando el gestor de paquetes `pip`:

- Para instalar **NumPy**:
  ```bash
  pip install numpy
  ```

- Para instalar **Matplotlib**:
  ```bash
  pip install matplotlib
  ```

- Para instalar **OpenCV**:
  ```bash
  pip install opencv-contrib-python
  ```

### Ejecución del Proyecto

Después de seguir estos pasos, ya estarás listo para ejecutar el código y procesar la imagen de prueba. Asegúrate de tener las imágenes disponibles en el mismo directorio que tu script o de modificar el código para apuntar a las rutas correctas.

Para esto basta con ejecutar el archivo .py del problema de su preferencia.


## Contribuciones

Si deseas contribuir a este proyecto, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza los cambios y haz commit (`git commit -m 'Agrega nueva funcionalidad'`).
4. Sube los cambios a tu repositorio (`git push origin feature/nueva-funcionalidad`).
5. Crea un pull request para revisar tus cambios.

## Licencia

Este proyecto está bajo la licencia MIT.
def clean_csv(input_file, output_file):
    with open(input_file, 'rb') as f_in:
        content = f_in.read()
    
    # Decodificar reemplazando errores y eliminar caracteres no deseados
    cleaned = content.decode('utf-8', errors='replace')\
                   .encode('utf-8')\
                   .replace(b'\x80', b'')\
                   .replace(b'\ufffd', b'')
    
    with open(output_file, 'wb') as f_out:
        f_out.write(cleaned)

# Uso:
clean_csv('./los-angeles1-OCR.csv', './cleaned.csv')
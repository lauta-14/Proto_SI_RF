import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Datos de ejemplo (reemplazar con datos reales)
data = {
    'Forma de hoja': ['Ovalada', 'Lanceolada', 'Lobulada', 'Acicular', 'Ovalada', 'Lanceolada', 'Lobulada'],
    'Textura de corteza': ['Lisa', 'Rugosa', 'Fisurada', 'Escamosa', 'Lisa', 'Rugosa', 'Fisurada'],
    'Tipo de fruto': ['Cono', 'Nuez', 'Baya', 'Cono', 'Cono', 'Nuez', 'Baya'],
    'Especie': ['Pino', 'Roble', 'Arce', 'Abeto', 'Pino', 'Roble', 'Arce']
}

df = pd.DataFrame(data)

# Codificar las características categóricas
X = pd.get_dummies(df.drop('Especie', axis=1))
y = df['Especie']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
print(classification_report(y_test, y_pred))

# Ejemplo de uso para clasificar un nuevo árbol
nuevo_arbol = pd.DataFrame({
    'Forma de hoja': ['Lanceolada'],
    'Textura de corteza': ['Rugosa'],
    'Tipo de fruto': ['Nuez']
})

nuevo_arbol = pd.get_dummies(nuevo_arbol)

# Asegurarse de que nuevo_arbol tenga las mismas columnas que X_train
nuevo_arbol = nuevo_arbol.reindex(columns = X_train.columns, fill_value=0)

prediccion = model.predict(nuevo_arbol)
print("La especie predicha del nuevo árbol es:", prediccion[0])
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

class CSVInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Interface CSV")
        
        self.file_paths = []
        self.model = None
        self.encoder = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Criar widgets para seleção de arquivos e controle de ações
        self.file_label = ttk.Label(self.root, text="Selecione o(s) arquivo(s) CSV:")
        self.file_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.select_button = ttk.Button(self.root, text="Selecionar arquivo", command=self.load_file)
        self.select_button.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.train_button = ttk.Button(self.root, text="Treinar modelo", command=self.train_model)
        self.train_button.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.predict_button = ttk.Button(self.root, text="Prever próximos jogos", command=self.predict_next_games)
        self.predict_button.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
    def load_file(self):
        files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if files:
            self.file_paths = list(files)
        
    def train_model(self):
        if not self.file_paths:
            print("Nenhum arquivo selecionado.")
            return
        
        dados = pd.concat((pd.read_csv(file_path, sep=';') for file_path in self.file_paths), ignore_index=True)
        
        # Verifique se a coluna 'Resultado' existe
        if not 'Resultado' in dados.columns:
            print("Coluna 'Resultado' não encontrada nos dados.")
            return
        
        # Codificar a coluna 'Resultado' para preparar para o modelo
        self.encoder = LabelEncoder()
        dados['Resultado'] = self.encoder.fit_transform(dados['Resultado'])
        
        # Separar os dados para treino e teste
        X = dados.drop('Resultado', axis=1)
        y = dados['Resultado']
        
        # Transformar dados categóricos in dummies
        X = pd.get_dummies(X)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treinar o modelo
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Avaliar o modelo
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print("Modelo treinado com sucesso.")
        print(f"Acurácia: {accuracy:.2%}, Precisão: {precision:.2%}, Recall: {recall:.2%}, F1-score: {f1:.2%}")
        
    def predict_next_games(self):
        if not self.model:
            print("O modelo ainda não foi treinado. Por favor, treine o modelo primeiro.")
            return
        
        # Criar tabela de resultados se não existir
        if not hasattr(self, 'result_table'):
            self.result_table = ttk.Treeview(self.root, columns=[
                "Data", "Horário", "Rodada", "Oponente", 
                "Formação com chances de Vitória", "Previsão", 
                "Precisão (%)"
            ], show="headings")
            
            self.result_table.heading("Data", text="Data")
            self.result_table.heading("Horário", text="Horário")
            self.result_table.heading("Rodada", text="Rodada")
            self.result_table.heading("Oponente", text="Oponente")
            self.result_table.heading("Formação com chances de Vitória", text="Formação com chances de Vitória")
            self.result_table.heading("Previsão", text="Previsão")
            self.result_table.heading("Precisão (%)", text="Precisão (%)")
            
            self.result_table.column("Data", width=80, anchor=tk.CENTER)
            self.result_table.column("Horário", width=80, anchor=tk.CENTER)
            self.result_table.column("Rodada", width=80, anchor=tk.CENTER)
            self.result_table.column("Oponente", width=120, anchor=tk.CENTER)
            self.result_table.column("Formação com chances de Vitória", width=160, anchor=tk.CENTER)
            self.result_table.column("Previsão", width=80, anchor=tk.CENTER)
            self.result_table.column("Precisão (%)", width=100, anchor=tk.CENTER)
            
            self.result_table.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Limpar tabela antes de adicionar novas previsões
        self.result_table.delete(*self.result_table.get_children())
        
        # Fazer previsões para cada arquivo selecionado
        for file_path in self.file_paths:
            dados = pd.read_csv(file_path, sep=';')
            
            if 'Oponente' in dados.columns:
                for index, row in dados.iterrows():
                    if pd.isna(row['Resultado']):  # Verifica se a previsão é necessária
                        # Criar dados para previsão
                        X_pred = pd.get_dummies(row.drop('Resultado'))
                        
                        # Adicionar colunas ausentes como zeros para compatibilidade com o modelo
                        missing_columns = list(set(self.X_train.columns) - set(X_pred.columns))
                        for col in missing_columns:
                            X_pred[col] = 0
                        
                        # Alinhar colunas para garantir consistência
                        X_pred = X_pred[self.X_train.columns]
                        
                        # Fazer previsão
                        prediction_probabilities = self.model.predict_proba(X_pred)[0]
                        
                        # Obter a previsão com maior probabilidade
                        predicted_class = self.encoder.inverse_transform([prediction_probabilities.argmax()])[0]
                        max_probability = max(prediction_probabilities) * 100  # Convertendo para porcentagem
                        
                        # Adicionar na tabela de resultados
                        self.result_table.insert("", "end", values=[
                            row.get("Data", "N/A"), row.get("Horário", "N/A"), row.get("Rodada", "N/A"), 
                            row["Oponente"], 
                            "3-4-3",  # Assumindo uma formação padrão, se necessário
                            predicted_class, 
                            f"{max_probability:.2f}"
                        ])

root = tk.Tk()
app = CSVInterface(root)
root.mainloop()

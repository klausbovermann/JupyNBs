import pandas as pd 
import numpy as np 
from graphviz import Digraph

styles = {
    
    'leaf': {'shape': 'rect', 'style': 'filled', 'color': 'yellow'},
    'value': {'shape': 'rect', 'style': 'filled', 'color': 'lightblue', 'height': '0.0001'},
    'crit': {'shape': 'rect'},
    'edge': {'labelangle': '0.0', 'labeldistance': '5.0', 'style': 'dotted'}
}


class split:
    #Initialisierung des Splits
    def __init__(self, attribute, values , split_type, bound = None):
        
        self.split_attribute = attribute      # Name des Attributes  
        self.split_values = values            # Menge der Split values 
        self.split_type = split_type          # Einer der Werte 'categorical' oder 'numerical'
        self.split_bound = bound              # Wird nur angegeben  wenn  split_type == numerical 
        pass
     
    def status(self):
        print('\n Attribut:',self.split_attribute,'\n split_values:', self.split_values)
        pass
    
    def copy(self):
        copy_split = split(self.split_attribute,self.split_values.copy(), self.split_type, self.split_bound )
        
        return copy_split
        pass
    
    pass

class node:
     #Initialisierung des Nodes
    def __init__(self, nNr = None, nLabel = None, nType = None, data = None, n_split = None):
        
        self.node_nr = nNr               # Nummer des Knotens
        self.node_label = nLabel         # Beschriftung des Knotens
        self.node_type = nType           # eine der Ausprägungen 'criterion'/'leaf'     
        self.subset = data               # Teildatensatz der Trainingsdaten
        self.node_split = n_split        # 'categorical' oder 'numerical' (nur angeben, wenn node_type = 'criterion')
        
        self.parent = None
        self.child_nodes = []
        self.edges = []
        
    def status(self):
        children = []
        for i in self.child_nodes:
            children.append(i.node_nr)
            
        print('\n Nr:',self.node_nr, '\n Label:', self.node_label,'\n Type:', self.node_type, '\n Children:', children, '\n Data: \n', self.subset )
        
    def copy(self):
        
        copy_node = node(self.node_nr, self.node_label, self.node_type, self.subset.copy(), self.node_split)
        copy_node.parent = self.parent
        
        for nd in self.child_nodes:
            copy_node.child_nodes.append(nd.copy())
            copy_node.child_nodes[-1].parent = copy_node
        
        for ed in self.edges:
            copy_node.edges.append(ed.copy()) 
        
        return copy_node
    pass

class edge:
    #Initialisierung 
    def __init__(self, root_nr = None, target_nr = None, label = ''):
        self.root_nr = root_nr
        self.target_nr = target_nr       
        self.label = label     
    
    def status(self):
        print('\n Root:',self.root_nr,'\n Target:', self.target_nr, '\n Label:', self.label)
        
    def copy(self):
        copy_edge = edge(self.root_nr, self.target_nr, self.label)
        
        return copy_edge
    
    pass

def find_all_splits(df_inputs, target_variable):
    
    # Wir wollen alle möglichen Splits in einer Liste sammeln
    # Für jedes Attribut wird ein Split erstellt und in list_of_splits abgespeichert
    list_of_splits = [] 
    
    attribute_list_categorical = df_inputs.drop(target_variable, axis = 1).select_dtypes(exclude = 'number').columns
    attribute_list_numerical = df_inputs.drop(target_variable, axis = 1).select_dtypes(include = 'number').columns
    
    
    # Wir gehen der Reihe nach alle kategorialen Attribute durch
    for current_attribute in attribute_list_categorical:            
        
        # Die verschiedenen Werte in der Spalte des aktuellen Attributs... 
        # ... werden mittels der unique()-Funktion ausgelesen
        value_set = df_inputs[current_attribute].dropna().unique()                                                      
        
        # Der aktuelle Split wird erstellt und er besteht aus Split-Attribut und Wertemenge 
        current_split = split(current_attribute, value_set, 'categorical') 
        
        list_of_splits.append(current_split)
        
        
     # Wir gehen der Reihe nach alle numerischen Attribute durch
    for current_attribute in attribute_list_numerical:            
    
        value_set = df_inputs[current_attribute].dropna().sort_values().unique()                                                      
        
        for i in range(len(value_set) - 1):
            current_bound = (value_set[i] + value_set[i+1]) / 2
            split_values = ['≤' + str(round(current_bound,2)), '>' + str(round(current_bound,2)) ]
            current_split = split(current_attribute, split_values, 'numerical', current_bound) 
            list_of_splits.append(current_split)        
                    
          
    return list_of_splits
    #Output: List of splits in the form --> 1 Split: [split_attribute, split_values]
    pass

def distribution(df_input, target_variable):
    
    return df_input[target_variable].value_counts().sort_index().tolist()

def loss_of_dispersion(df_inputs, target_variable, current_split):
    
    pre_dispersion = df_inputs[target_variable].var(ddof=0)
    post_dispersion = 0
    
    total_elements = len(df_inputs)
    
    if current_split.split_type == 'categorical':
        for split_value in current_split.split_values:

            current_df = df_inputs[df_inputs[current_split.split_attribute] == split_value]
            elements = len(current_df)
            post_dispersion += (elements/total_elements)*current_df[target_variable].var(ddof=0)
   
    elif current_split.split_type == 'numerical': 
        
        subset1 = df_inputs[df_inputs[current_split.split_attribute] <= current_split.split_bound]
        elements = len(subset1)
        post_dispersion += (elements/total_elements)*subset1[target_variable].var(ddof=0)
            
        subset2 = df_inputs[df_inputs[current_split.split_attribute] > current_split.split_bound]
        elements = len(subset2)
        post_dispersion += (elements/total_elements)*subset2[target_variable].var(ddof=0)
        
    else:
        post_dispersion = pre_dispersion
    
    
    loss_of_disp = pre_dispersion - post_dispersion
    
    return loss_of_disp

        

def identify_best_split(df_inputs, target_variable, list_of_splits):
    
    best_split = None
    best_information_gain = -1
    
    for current_split in list_of_splits:
            
            current_information_gain = information_gain(df_inputs, target_variable, current_split)
            
            if current_information_gain >= best_information_gain:
                best_information_gain = current_information_gain
                best_split = current_split
            
    return best_split
    #Output: Most productive split according to information gain
    pass

def identify_best_split_regr(df_inputs, target_variable, list_of_splits):
    
    best_split = None
    best_loss_of_dispersion = -1
    
    for current_split in list_of_splits:
            current_loss_of_dispersion = loss_of_dispersion(df_inputs, target_variable, current_split)
            
            if current_loss_of_dispersion >= best_loss_of_dispersion:
                best_loss_of_dispersion = current_loss_of_dispersion
                best_split = current_split
            
    
    return best_split
    #Output: Most productive split according to information gain
    pass

def apply_split(df_inputs, current_split):
    
    list_of_subsets = []
    
    if current_split.split_type == 'categorical':
        for current_value in current_split.split_values:

            current_df = df_inputs[df_inputs[current_split.split_attribute] == current_value]
            #current_df.append(df_inputs[df_inputs[current_value].isna()])                          #Umgang mit NaN einfach in jeden Ast weiterleiten
            list_of_subsets.append(current_df)
            
    elif current_split.split_type == 'numerical':
        subset1 = df_inputs[df_inputs[current_split.split_attribute] <= current_split.split_bound]
        list_of_subsets.append(subset1)
            
        subset2 = df_inputs[df_inputs[current_split.split_attribute] > current_split.split_bound]
        list_of_subsets.append(subset2)
            
    
    return list_of_subsets
    #Output: List of Subsets(DataFrames)
    pass

class RegressionTree:
    
    #Initialisierung des Decision Tree
    def __init__(self):
        
        self.tree_edges = []
        self.tree_nodes = []
        self.tree_graph = Digraph()
        self.target = None
        self.root = None
        #self.attributes = None
        
        pass
    
    
    def fit_tree(self, df_input, target_variable):
        
        # Fehlende Werte für die Zielvariable droppen?
        
        self.target = target_variable
        attributes = (df_input.columns).drop(target_variable)
        print('loading...')
    
        # Überprüfen, ob mehr als ein Wert für die Zielvariable vorliegt
        if df_input[target_variable].nunique() == 1:
            
            # Falls die Werte der Zielvariable im Datensatz alle gleich sind gib Leaf-Knoten mit dem Wert aus 
            label = df_input[target_variable].mean()
            self.return_leaf_node(label, df_input)

            pass
        
        # Überprüfen ob noch Attribute für weitere Splits übrig sind
        elif len(attributes) == 0:
            
            #Falls Anzahl der Attribute 0 ist, gib ein leaf mit dem Mehrheitswert aus
            label = df_input[target_variable].mean()
            self.return_leaf_node(label, df_input)
            
            pass
        
        # Falls vorherige Abfragen nicht zutrafen wird ein weiterer Split gesucht um ihn anzuwenden
        else:
            #Finde alle möglichen Splits
            list_of_splits = find_all_splits(df_input, target_variable)
            
            if len(list_of_splits) != 0:
                
                #Identifiziere den besten Split unter allen Splits
                best_split = identify_best_split_regr(df_input, target_variable, list_of_splits)


                # Überprüfen: Ist der Split produktiv?
                if (loss_of_dispersion(df_input, target_variable, best_split) > 0):
                    
                    #Wende den besten Split auf die Inputdaten an und erstelle somit ein Liste von Teildatensätzen
                    list_of_subsets = apply_split(df_input, best_split)                                  

                                                      
                    #Den erstellten Split als Knoten ausgeben, falls best_split produktiv ist        
                    self.return_split_node(best_split, df_input)
                    current_node = self.tree_nodes[-1]


                    #Rekursive weitere Anwendung für jeden erstellten Teildatensatz
                    for i in range(len(list_of_subsets)):
                        
                        next_node_nr = str(len(self.tree_nodes) + 1)                               

                        new_input_subset = list_of_subsets[i].drop(best_split.split_attribute, axis = 1)

                        self.fit_tree(new_input_subset, target_variable)
                       
                        self.new_edge(root = current_node.node_nr, target = next_node_nr, label = best_split.split_values[i])# N E U

                else:
                    #Falls best_split nicht produktiv, dann leaf ausgeben
                    label = df_input[target_variable].mean()
                    self.return_leaf_node(label, df_input)
                    pass
                
            else:
                #Falls list_of_splits leer ist gib einen 'leaf-node' aus
                label = df_input[target_variable].mean()
                self.return_leaf_node(label, df_input)
                pass    
    
        pass
    
    
    def return_leaf_node(self, node_label, data):
        
        node_nr = str(len(self.tree_nodes) + 1)# Neuer Knoten bekommt die nächst freie Nummer in tree_nodes
        
        current_node = node(node_nr, round(float(node_label),2), 'leaf', data) # Knoten wird erstellt
        
        self.tree_nodes.append(current_node) # Knoten wird zu Liste aller Knoten hinzugefügt
    
    
    def return_split_node(self, best_split, data):
        
        node_nr = str(len(self.tree_nodes) + 1)       # Neuer Knoten bekommt die nächst freie Nummer in tree_nodes
        node_label = best_split.split_attribute       # Das Label des Knotens ist das aktuelle Split Attribut
        current_node = node(node_nr, node_label, 'criterion', data, best_split) # Knoten wird erstellt
        self.tree_nodes.append(current_node)                        # Knoten wird zu Liste aller Knoten hinzugefügt
        
    
    def new_edge(self, root, target, label):
        
        #Edge bauen
        new_edge = edge(root, target, str(label))                                            # N E U
        self.tree_edges.append(new_edge)
        
        
        #Nodes informieren
        nd_root = self.tree_nodes[int(root) - 1]                                  # gegebenenfalls verbessern?
        nd_target = self.tree_nodes[int(target) - 1]
            
        nd_root.child_nodes.append(nd_target)
        nd_root.edges.append(new_edge)
        
        nd_target.parent = nd_root
        nd_target.edges.append(new_edge)
    
    
    def query(self, input_series):
        #Input: Series, die einen Wert für jedes Attribut + Zielvariable enthält
        
        current_node = self.tree_nodes[0]
        next_nr = current_node.node_nr
        
        # Wir gehen so lange durch den Baum, bis wir in einem 'leaf-node' sind
        while current_node.node_type == 'criterion':
            
            #Prüfwert um später zu schauen, ob eine neuer Knoten gefunden wurde
            old_nr = next_nr
            
            #prüfen ob der Split 'categorical' oder 'numerical' ist
            if current_node.node_split.split_type == 'categorical':
    
            # Suche die Kante, die am aktuellen Knoten liegt und zum Wert der Input Series passt
                for edge in current_node.edges: 
                    if (edge.label == str(input_series[current_node.node_label])):
                        next_nr = edge.target_nr
                        break
                
                #Prüfen ob neuer Knoten gefunden wurde
                #Falls der im Kriterum abgefragte Wert fehlt (NaN) kann kein Ast ausgewählt werden
                #der Baum gibt den Mittelwert im aktuellen Kriteriums-Knoten aus 
                if old_nr == next_nr:
                    return current_node.subset[self.target].mean()
                
                # der nächste Knoten wir gesucht und als current_node gespeichert, um in die nächste Iteration der Schleife zu gehen 
                for nd in current_node.child_nodes:
                    if nd.node_nr == next_nr:
                        current_node = nd
                        break
                        
            elif current_node.node_split.split_type == 'numerical':
                
                if (input_series[current_node.node_label] <= current_node.node_split.split_bound):
                    current_node = current_node.child_nodes[0]
                else:
                    current_node = current_node.child_nodes[1]
                    
        return current_node.node_label

        pass

    
    
    def prediction_accuracy(self, df_input):
        
        targets = df_input[self.target]

        prediction_list = []
        
        for i in range(len(df_input)):
            prediction_list.append(self.query(df_input.iloc[i]))
        
        predictions = pd.Series(prediction_list)
        predictions.index = targets.index
        mean_deviation = (abs(predictions - targets)).mean()
        #dispersion = (abs(predictions - targets)).mad()
        #max_deviation = (abs(predictions - targets)).max()
        return mean_deviation#, max_deviation, dispersion
        
        pass  
    
    
    def prune_node(self, prune_node_nr, prune_node = None):
        
        if prune_node == None:
            for nd in self.tree_nodes:
                if nd.node_nr == prune_node_nr:
                    prune_node = nd
                    break
                    
        if prune_node.parent == None:
            root_node = node()
        else:
            root_node = prune_node.parent
        
        list_of_children = prune_node.child_nodes
        prune_node.child_nodes = []
                
        if (prune_node.node_type == 'leaf') & (root_node.node_type == 'leaf'):
            
            self.tree_nodes.remove(prune_node)
            
            for edge in prune_node.edges:
                if edge.target_nr == prune_node_nr:
                    self.tree_edges.remove(edge)
                    #root_node.edges.remove(edge)                                        # Warum geht das nicht so?
                    for ed in root_node.edges:
                        if ed.target_nr == prune_node_nr:
                            root_node.edges.remove(ed)
        
        else:
            prune_node.node_type = 'leaf'
            prune_node.node_label = prune_node.subset[self.target].mean()
            
            
            
            
            
            
            for child in list_of_children:
                self.prune_node(child.node_nr, child)
            
            
            if root_node.node_type == 'leaf':
                self.tree_nodes.remove(prune_node)
    
                for edge in prune_node.edges:
                    if edge.target_nr == prune_node_nr:
                        self.tree_edges.remove(edge)
                        #root_node.edges.remove(edge)
                        for ed in root_node.edges:
                            if ed.target_nr == prune_node_nr:
                                root_node.edges.remove(ed)
   
        pass
    

    

    def validation_pruning(self, validation_sample, root_node = None):       #root_nr = 1
        
        if root_node == None:
            current_node = self.tree_nodes[0]
        else:
            current_node = root_node
        
        if current_node.node_type == 'leaf':
            
            pass

        else:  
            for child in current_node.child_nodes:                                                                    
                test_tree = self.validation_pruning_regr(validation_sample, child)

            print('Test Node:', current_node.node_nr)
            
            if test_tree == None:
                test_tree = self.copy()
            
            pre_test_deviation = test_tree.prediction_accuracy_regr(validation_sample)

            test_tree.prune_node(current_node.node_nr)

            post_test_deviation = test_tree.prediction_accuracy_regr(validation_sample)
             
                
            if post_test_deviation <= pre_test_deviation:
                self.prune_node(current_node.node_nr, current_node)
                print('Prune Node:', current_node.node_nr)
                print('Node-Count', len(self.tree_nodes))
                print('New Test-Deviation', post_test_deviation)
                return test_tree
            
            return None
        pass
    

    
    def print_tree(self):
        
        self.tree_graph = Digraph(filename = ('Tree_' + self.target))
        for current_node in self.tree_nodes:
            if current_node.node_type == 'criterion':
                self.tree_graph.node(current_node.node_nr, str(current_node.node_label) + '?' + '\n Nr.:' + current_node.node_nr + '\n' + '\n' + 'Count: '+ str(current_node.subset[self.target].count()) + '\n'+ 'Mean: ' + str(round(current_node.subset[self.target].mean(), 2)) + '\n'+ 'MSE: ' + str(round(current_node.subset[self.target].var(ddof = 0), 2)), styles['crit'])
            elif current_node.node_type == 'leaf':
                self.tree_graph.node(current_node.node_nr, str(current_node.node_label) + '\n Nr.:' + current_node.node_nr + '\n' + '\n' + 'Count: '+ str(current_node.subset[self.target].count())+ '\n'+ 'MSE: ' + str(round(current_node.subset[self.target].var(ddof = 0), 2)), styles['leaf'])    

        for current_edge in self.tree_edges:
                self.tree_graph.edge(current_edge.root_nr, current_edge.target_nr, current_edge.label)

        self.tree_graph.view()
        pass
        
        
        
    def copy(self, copy_tree = None, current_node = None):
        
        if copy_tree == None:
            copy_tree = RegressionTree()
            copy_tree.target = self.target
            current_node = self.tree_nodes[0].copy()
        
        copy_tree.tree_nodes.append(current_node)
        for edge in current_node.edges:
            if edge.target_nr == current_node.node_nr:
                copy_tree.tree_edges.append(edge)
                
        if len(current_node.child_nodes) == 0:
            pass
        else:
            for nd in current_node.child_nodes:
                copy_tree = self.copy(copy_tree, nd)
        
        
        return copy_tree
        pass
        
    
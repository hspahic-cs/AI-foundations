from matplotlib.backend_bases import MouseButton
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

class Data_Sample:
    """Class for making a quick and dirty dataset to test Linear Classifiers. 

    Creates an empty 2D plot, adding data points of class 0 at point clicked 
    by user. User can switch to following classes through right click. Once
    all classes added, runs model. 
    
    Methods:

    
    Attributes: 
        (int) num_classes: represents number of classifiers in dataset
        (int[]) data: holds each datapoint in form (x_cord, y_cord, class)
        (int) status: represents class currently being added
    """

    def __init__ (self, num_classes, data=[]):
        """Inits dataset to add data from first class (status = 0).
        NOTE: Allows user to add their own default data."""

        self.num_classes = num_classes
        self.data = data
        self.status = 0
        
        # Create plot
        fig, ax = plt.subplots()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title('Please enter points below')

        self.fig, self.ax = fig, ax
        self.connections = ()

    def connect(self):
        """Binds events to plot!
        NOTE: Currently only binds on_click. Designed for future
        functionality in mind
        """
        self.connections = (
            self.fig.canvas.mpl_connect('button_press_event', self.on_click),
        )

    def disconnect(self):
        """Disconnects events from plot!
        """
        for connection in self.connections:
            self.fig.canvas.mpl_disconnect(connection)
        print("Disconnecting")
    
    def on_click(self, event):
        '''Records & plots data-points where clicked!
        On left click: records & plots point of current class
        On right click: moves to subsequent class

        TODO: Generalize upto n classes. Currently only supports 8.
        '''

        colors = cm.get_cmap('Set1', 8)

        # Changes classes on right click, exits on last
        if event.button is MouseButton.RIGHT:
            if(self.status < self.num_classes):
                self.status += 1
            else:
                self.disconnect()
                plt.close()

            return

        # Records data point & plots result
        if event.inaxes:
            self.data.append([event.xdata,event.ydata])
            
            if(self.status < self.num_classes):
                self.data[-1].append(self.status)
                plt.scatter(self.data[-1][0], self.data[-1][1], marker="o", color=colors(self.status)) 
            else:
                print("Error: on_click still listening after closed")   

            plt.show()

    def run(self):
        self.connect()
        plt.show()
        self.data = np.asarray(self.data)

if __name__ == "__main__":
    x = Data_Sample(6)
    x.run()
import numpy as np

class SquatAnalyzer:
    """
    Analizzatore per la tecnica dello squat basato sulla traiettoria del bilanciere
    rispetto alla posizione del piede.
    """
    
    def __init__(self, safe_zone_px=50):
        """
        Inizializza l'analizzatore.
        
        Args:
            safe_zone_px: Dimensione della zona sicura in pixel intorno al midfoot
        """
        self.safe_zone_px = safe_zone_px
    
    def analyze_squat(self, trajectory_array, foot_position):
        """
        Analizza la tecnica dello squat confrontando la traiettoria del bilanciere
        con la posizione del midfoot.
        
        Args:
            barbell_trajectory: Array di punti (x, y) della traiettoria del bilanciere
            foot_position: Posizione del midfoot (x, y)
            
        Returns:
            Dictionary con metriche sull'esecuzione dello squat
        """
        
        # Calcola la deviazione orizzontale dal midfoot
        x_deviations = trajectory_array[:, 0] - foot_position[0]
        
        # Calcola statistiche sulla deviazione
        mean_deviation = np.mean(x_deviations)
        max_forward = np.max(x_deviations)    # Massima deviazione in avanti
        max_backward = np.min(x_deviations)   # Massima deviazione indietro
        std_deviation = np.std(x_deviations)  # Deviazione standard (stabilità)
        
        # Percentuale di tempo in cui il bilanciere è nella zona sicura
        in_safe_zone = np.abs(x_deviations) < self.safe_zone_px
        percent_in_safe_zone = np.mean(in_safe_zone) * 100
        
        # Identifica i punti di maggiore deviazione
        max_forward_idx = np.argmax(x_deviations)
        max_backward_idx = np.argmin(x_deviations)
        
        return {
            "mean_deviation_px": mean_deviation,
            "max_forward_deviation_px": max_forward,
            "max_backward_deviation_px": max_backward,
            "std_deviation_px": std_deviation,
            "percent_in_safe_zone": percent_in_safe_zone,
            "max_forward_point": tuple(trajectory_array[max_forward_idx]),
            "max_backward_point": tuple(trajectory_array[max_backward_idx]),
            "assessment": self._get_assessment(percent_in_safe_zone, max_forward, max_backward)
        }
    
    def _get_assessment(self, percent_in_safe_zone, max_forward, max_backward):
        """
        Fornisce una valutazione qualitativa della tecnica dello squat.
        
        Returns:
            Stringa con la valutazione
        """
        if percent_in_safe_zone > 90:
            return "Eccellente: Il bilanciere rimane sopra il midfoot per più del 90% dell'esercizio"
        elif percent_in_safe_zone > 80:
            return "Buono: Il bilanciere rimane sopra il midfoot per più dell'80% dell'esercizio"
        elif percent_in_safe_zone > 70:
            return "Accettabile: Il bilanciere rimane sopra il midfoot per più del 70% dell'esercizio"
        else:
            if max_forward > abs(max_backward):
                return f"Da migliorare: Il bilanciere tende a spostarsi troppo in avanti ({max_forward:.1f}px)"
            else:
                return f"Da migliorare: Il bilanciere tende a spostarsi troppo indietro ({abs(max_backward):.1f}px)"
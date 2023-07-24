"""Berechnung eines Tilgungsplans für ein Darlehen."""

# Anfangsschulden in Euro.
schulden = 350000

# Zinssatz in Prozent.
zinssatz = 1.9

# Monatliche Rate in Euro.
rate = 1800

# Zu Beginn sind noch keine Zahlungen geleistet worden.
monat = 0
zinsen_gesamt = 0
zahlungen_gesamt = 0

# Lass den Tilgungsplan enden, wenn der Kredit komplett
# zurückbezahlt wurde.
while schulden > 0:
    monat += 1
    # In jedem Monat fällt ein Zwölftel des Jahreszinssatzes an.
    # Runde das Ergebnis wird auf volle Cent und erhöhe
    # die Schulden.
    zins = round(schulden * zinssatz / 100 / 12, 2)
    schulden += zins

    # Es wird eine Zahlung in Höhe der Rate geleistet. Es wird
    # jedoch auf keinen Fall mehr als die verbleibende Restschuld
    # bezahlt.
    zahlung = min(rate, schulden)

    # Verringere die Schulden aufgrund der Zahlung.
    schulden -= zahlung

    # Summiere die Zahlungen und die gezahlten Zinsen für die
    # Zusammenfassung am Ende.
    zahlungen_gesamt += zahlung
    zinsen_gesamt += zins

    # Gib für diesen Monat eine Zeile im Tilgungsplan aus.
    print(f'{monat:3d}. Monat: Zinsen{zins:8.2f} €, '
          f'Tilgung{zahlung-zins:8.2f} €, '
          f'Restschuld{schulden:10.2f} €')

# Gib eine Zusammenfassung des Tilgungsverlaufs aus.
tilgungen_gesamt = zahlungen_gesamt - zinsen_gesamt
print()
print(f'Das Darlehen wurde nach {monat//12} Jahren und '
      f'{monat % 12} Monaten vollständig zurückbezahlt.')
print()
print(f'Gesamtsumme       : {zahlungen_gesamt:10.2f} €')
print(f'   davon Zinsen   : {zinsen_gesamt:10.2f} €')
print(f'   davon Tilgungen: {tilgungen_gesamt:10.2f} €')

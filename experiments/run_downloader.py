import sys
from src.data_loader import download_stock_data

def main(list_ticker: list[str], start_date: str, end_date: str):
    for ticker in list_ticker:
        data = download_stock_data(
            ticker=ticker, 
            start_date=start_date, 
            end_date=end_date
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 3:
        print("Usage: python run_downloader.py <list_ticker> <start_date> <end_date> (e.g. python -m experiments.run_downloader BBCA.JK,UNTR.JK,ASII.JK,TLKM.JK 2025-07-01 2025-07-31)")
        sys.exit(1)    

    list_ticker = args[0].split(",")
    start_date = args[1]
    end_date = args[2]
    
    
    main(list_ticker, start_date, end_date)






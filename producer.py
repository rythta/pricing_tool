import redis
import time
import sys

def produce_items(item):
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.rpush('queue', item)  # rpush is used to insert the item at the end of the list named 'queue'
    print(f"Pushed {item}")

if __name__ == "__main__":
    produce_items(sys.argv[1]+","+sys.argv[2]+","+sys.argv[3]+","+sys.argv[4])


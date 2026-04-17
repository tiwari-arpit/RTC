# import argparse
# from core.pipeline import RealtimePipeline

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--source", default=0)
#     parser.add_argument("--no-caption", action="store_true")
#     parser.add_argument("--save", default=None)
#     args = parser.parse_args()

#     source = int(args.source) if str(args.source).isdigit() else args.source

#     pipeline = RealtimePipeline(
#         video_source=source,
#         use_caption=not args.no_caption,
#         save_output=args.save
#     )
#     pipeline.run()
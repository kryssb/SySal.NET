using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using System.Xml.Serialization;

namespace SySal.Executables.PostProcessingManager
{
    partial class Program
    {
        static bool MainFeedData(bool isdoc, string [] args)
        {
            if (isdoc)
            {
                Console.WriteLine("/feed <.scan file path> <dest dir> [<first index> [<last index>]]");
                Console.WriteLine("Feeds data from the specified .scan path to the destination path, optionally restricting to a start and last index.");
                return true;
            }
            if (args.Length == 0 || string.Compare(args[0], "/feed", true) != 0) return false;
            string sourcescanfile = null;
            string destpath = null;
            try
            {
                sourcescanfile = args[1];
            }
            catch (Exception x)
            {
                throw new Exception("Cannot read the path of the source .scan file.", x);
            }
            try
            {
                destpath = args[2];
            }
            catch (Exception x)
            {
                throw new Exception("Cannot read the destination path.", x);
            }
            int firstindex = -1;
            int lastindex = -1;
            if (args.Length >= 4)
                try
                {
                    firstindex = (int)uint.Parse(args[3]);
                }
                catch (Exception x)
                {
                    throw new Exception("Cannot read the first index.", x);
                }
            if (args.Length >= 5)
                try
                {
                    lastindex = (int)uint.Parse(args[3]);
                }
                catch (Exception x)
                {
                    throw new Exception("Cannot read the last index.", x);
                }
            destpath = destpath.TrimEnd(new char[] { '/', ' ', '\\', '\t', '\n' });
            try
            {
                System.IO.Directory.CreateDirectory(destpath);
            }
            catch (Exception x)
            {
                throw new Exception("Cannot create path \"" + destpath + "\".", x);
            }
            Zone zs = Zone.FromFile(sourcescanfile);
            Zone zd = new Zone() { FileNameTemplate = zs.FileNameTemplate, AutoDelete = zs.AutoDelete };
            zs.AutoDelete = false;
            zd.ReplaceDirectory(destpath);            
            if (firstindex < 0)
                RetryCopy(zs.StartFileName, zd.StartFileName);
            uint index;
            if (firstindex < 0) firstindex = 0;
            if (lastindex < 0) lastindex = (int)zs.Strips - 1;
            for (index = (uint)firstindex; index <= lastindex; index++)
            {
                for (uint view = 0; view < zs.Views; view++)
                {
                    RetryCopy(zs.GetClusterFileName(index, true, view), zd.GetClusterFileName(index, true, view));
                    RetryCopy(zs.GetClusterFileName(index, false, view), zd.GetClusterFileName(index, false, view));
                }
                zd.Progress.BottomStripsReady = zd.Progress.TopStripsReady = index + 1;
                zd.UpdateWrite();
                Console.WriteLine("Copied strip " + index);
            }
            zd.AutoDelete = false;
            return true;
        }

        static void RetryCopy(string s, string d)
        {
            while (true)
                try
                {
                    if (System.IO.File.Exists(d))
                    {
                        System.IO.File.SetAttributes(d, System.IO.FileAttributes.Normal);
                        System.IO.File.Delete(d);
                    }
                    System.IO.File.Copy(s, d, true);
                    System.IO.File.SetAttributes(d, System.IO.FileAttributes.Normal);
                    return;
                }
                catch (Exception x)
                {
                    var fg = Console.ForegroundColor;
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Cannot copy \"" + s + "\" to \"" + d + "\": " + x.Message + " - Retrying soon");
                    Console.ForegroundColor = fg;
                    System.Threading.Thread.Sleep(5000);
                }
        }
    }
}
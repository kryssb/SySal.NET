#ifndef _XML_LITE_H_
#define _XML_LITE_H_

#include <string>
#include <locale>
#include <vector>

using namespace std;

namespace XMLLite
{
	class Tag
	{
	public:
		std::string Name;
		char *TagStart;
		char *TagEnd;
		bool AutoClose;
		bool Closing;		
		bool Empty() { return Name.empty(); }

		Tag() { Name = ""; AutoClose = false; Closing = false; TagStart = TagEnd = 0; }
		Tag(const char *str, int &nextpos)
		{
			Name = "";
			AutoClose = false;
			Closing = false;
			TagStart = TagEnd = 0;
			char *scan = (char *)str;
			while (*scan != 0 && *scan != '<') scan++;
			if (*scan == 0) 
			{
				nextpos = scan - str;
				return;
			}
			TagStart = scan;
			char *namestart = ++scan;
			while (*scan != 0 && *scan != '>') scan++;
			if (*scan == 0) throw "Malformed tag: missing closing '>'.";
			TagEnd = scan;
			nextpos = (scan - str + 1);
			char *nameend = scan;
			for (scan = namestart; scan < nameend && isspace(*scan); scan++);
			if (scan == nameend) throw "Empty tag.";
			if (*scan == '/')
			{
				AutoClose = false;
				Closing = true;

				scan++;
				while (scan < nameend && isspace(*scan)) scan++;
				if (scan == nameend) throw "Missing tag name.";
				while (scan < nameend && isspace(*scan) == false && *scan != '/')
				{
					Name += *scan++;
				}
				return;
			}
			else
			{
				while (scan < nameend && isspace(*scan) == false && *scan != '/')
				{
					Name += *scan++;
				}
				scan = nameend - 1;
				while (scan >= namestart && isspace(*scan)) scan--;
				if (scan >= namestart && *scan == '/')
				{
					AutoClose = true;
					Closing = true;
				}
				return;
			}
		}
	};

	class Element
	{
	protected:
		std::vector<Element> Elements;
	
	public:
		std::string Name;
		std::string Value;

		Element() { Name = ""; Value = ""; }

		Element &operator[](char *s)
		{ 
			vector<Element>::iterator iter = Elements.begin(); 
			while (iter != Elements.end()) 
			{
				if (iter->Name == s)
					return *iter;
				iter++;
			}
			static Element empty;
			empty.Name = "";
			empty.Value = "";
			return empty;
		}

		Element &operator[](int i)
		{
			return Elements[i];
		}

		int Size() { return Elements.size(); }

		vector<Element> &operator<<(Element &a) { Elements.push_back(a); return Elements; }

		std::string ToString(int indent = 0)
		{
			string intstring = "";
			int i;
			for (i = 0; i < indent; i++)
				intstring += " ";
			if (Elements.size() == 0)
			{
				return intstring + "<" + Name + ">" + Value + "</" + Name + ">\n";
			}
			string outstr = intstring + "<" + Name + ">\n";
			vector<Element>::iterator iter = Elements.begin();
			while (iter != Elements.end())
			{
				outstr += iter->ToString(indent + 1);
				iter++;
			}
			return outstr + intstring + "</" + Name + ">\n";
		}

		static Element FromTagVector(vector<Tag> &tags)
		{
			Tag &tag = tags[0];
			if (tag.AutoClose)
			{
				Element el;
				el.Name = tag.Name;
				el.Value = "";
				tags.erase(tags.begin(), tags.begin() + 1);
				return el;
			}
			if (tag.Closing) throw "Found closing tag without opening tag.";
			if (tags.size() == 1) throw "Missing closing tag.";
			int i;
			int lev = 1;
			for (i = 1; i < tags.size(); i++)
			{
				if (tags[i].AutoClose == false)
				{
					if (tags[i].Closing)
					{
						lev--; 
						if (lev == 0) break;
					}
					else lev++;
				}
			}
			if (lev != 0) throw "Missing closing tag.";
			if (tags[i].Name != tag.Name) throw "Mismatched closing tag.";
			Element el;
			el.Name = tag.Name;
			el.Value = "";
			vector<Tag> mytags;
			int j;
			char *tag_s = tag.TagEnd + 1;
			char *tag_e = tags[i].TagStart - 1;
			for (j = 1; j < i; j++) mytags.push_back(tags[j]);
			tags.erase(tags.begin(), tags.begin() + i + 1);
			while (mytags.size() > 0)			
				el.Elements.push_back(FromTagVector(mytags));
			if (el.Elements.size() == 0)			
				while (tag_s <= tag_e) 
					el.Value += *tag_s++;
			return el;			
		}

		static vector<Element> FromString(const char *str) 
		{
			int maxsize = strlen(str);
			std::vector<XMLLite::Tag> tags;
			int pos = 0;
			const char *scan = str;
			while (scan - str < maxsize)
			{
				XMLLite::Tag tag((const char *)scan, pos);
				scan += pos;
				if (tag.Empty() == false) 
				{
					tags.push_back(tag);
				}
			}
			if (tags.size() == 0) throw "No element found.";
			
			vector<Element> els;
			while (tags.size() > 0)			
				els.push_back(FromTagVector(tags));
			return els;
		}
	};
}

#endif
#include "apue.h"
#include <dirent.h>
#include <limits.h>

/* function type that is called for each filename */
typedef it Myfunc(const char*, const struct static)

int
main(int argc, char *argv[])
{
	int		ret;

	if (argc != 2)
		err_quit("usage:  ftw  <starting-pathname>");


		ntot = 1;		/* avoid divide by 0; print 0 for all counts */
	printf("regular files  = %7ld, %5.2f %%\n", nreg,
	  nreg*100.0/ntot);
	printf("directories    = %7ld, %5.2f %%\n", ndir,
	  ndir*100.0/ntot);
	printf("block special  = %7ld, %5.2f %%\n", nblk,
	  nblk*100.0/ntot);
	printf("char special   = %7ld, %5.2f %%\n", nchr,
	  nchr*100.0/ntot);
	printf("FIFOs          = %7ld, %5.2f %%\n", nfifo,
	  nfifo*100.0/ntot);
	printf("symbolic links = %7ld, %5.2f %%\n", nslink,
	  nslink*100.0/ntot);
	printf("sockets        = %7ld, %5.2f %%\n", nsock,
	  nsock*100.0/ntot);
	exit(ret);
}

/*
 * Descend through the hierarchy, starting at "pathname".
 * The caller's func() is called for every file.
 */
		/* file other than directory */
		/* directory */
		/* directory that can't be read */
		/* file that we can't stat */

		/* contains full pathname for every file */


static int					/* we return whatever func() returns */
myftw(char *pathname, Myfunc *func)
{
	/* malloc path_max+1 bytes */
										/* ({Prog pathalloc}) */
			err_sys("realloc failed");
	}
}

/*
 * Descend through the hierarchy, starting at "fullpath".
 * If "fullpath" is anything other than a directory, we lstat() it,
 * call func(), and return.  For a directory, we call ourself
 * recursively for each name in the directory.
 */
static int					/* we return whatever func() returns */
dopath(Myfunc* func)
{


	/*
	 * It's a directory.  First call func() for the directory,
	 * then process each filename in the directory.
	 */
		return(ret);

		pathlen *= 2;
			err_sys("realloc failed");
	}
		return(func(fullpath, &statbuf, FTW_DNR));

	while ((dirp = readdir(dp)) != NULL) {
				continue;		/* ignore dot and dot-dot */
			break;	/* time to leave */
	}

		err_ret("can't close directory %s", fullpath);
	return(ret);
}

static int
myfunc(const char *pathname, const struct stat *statptr, int type)
{
	switch (type) {
			err_dump("for S_IFDIR for %s", pathname);
		}
		break;
		ndir++;
		break;
		err_ret("can't read directory %s", pathname);
		break;
		err_ret("stat error for %s", pathname);
		break;
		err_dump("unknown type %d for pathname %s", type, pathname);
	}
	return(0);
}

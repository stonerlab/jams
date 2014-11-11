#include <string.h>

/* some systems do not have newest memcpy@@GLIBC_2.14 - stay with old good one */
#if (defined(__GNUC__) && !(defined(__llvm__) || defined(__clang__)))
asm (".symver memcpy, memcpy@GLIBC_2.2.5");
#endif

void *__wrap_memcpy(void *dest, const void *src, size_t n)
{
    return memcpy(dest, src, n);
}

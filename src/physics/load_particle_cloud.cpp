#include "physics/load_particle_cloud.h"

#include "common/logger.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace cressim::neo::physics
{

namespace
{

constexpr std::uintmax_t kCountPrefixBytes = sizeof(std::uint32_t);
constexpr std::uintmax_t kParticleBytes    = sizeof(float) * 3u;

std::uint32_t readUint32LittleEndian(const unsigned char *bytes) noexcept
{
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8u) |
           (static_cast<std::uint32_t>(bytes[2]) << 16u) |
           (static_cast<std::uint32_t>(bytes[3]) << 24u);
}

} // namespace

bool readParticleCloudBin(const std::filesystem::path &path,
                          std::vector<Diligent::float3> &particles, std::string &errorMessage)
{
    particles.clear();
    errorMessage.clear();

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
    {
        std::ostringstream message;
        message << "Failed to open particle cloud binary file: " << path.string();
        errorMessage = message.str();
        return false;
    }

    const std::streampos endPosition = stream.tellg();
    if (endPosition < 0)
    {
        std::ostringstream message;
        message << "Failed to query particle cloud binary size: " << path.string();
        errorMessage = message.str();
        return false;
    }

    const std::uintmax_t fileSize = static_cast<std::uintmax_t>(endPosition);
    if (fileSize == 0u)
    {
        std::ostringstream message;
        message << "Particle cloud binary file is empty: " << path.string();
        errorMessage = message.str();
        return false;
    }

    stream.seekg(0, std::ios::beg);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(fileSize));
    if (!stream.read(reinterpret_cast<char *>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size())))
    {
        std::ostringstream message;
        message << "Failed to read particle cloud binary file: " << path.string();
        errorMessage = message.str();
        return false;
    }

    std::uintmax_t dataOffset    = 0u;
    std::uintmax_t particleCount = 0u;
    if (fileSize >= kCountPrefixBytes)
    {
        const std::uint32_t prefixedCount = readUint32LittleEndian(bytes.data());
        const std::uintmax_t countedSize =
            kCountPrefixBytes + static_cast<std::uintmax_t>(prefixedCount) * kParticleBytes;
        if (countedSize == fileSize)
        {
            dataOffset    = kCountPrefixBytes;
            particleCount = prefixedCount;
        }
    }

    if (dataOffset == 0u)
    {
        if ((fileSize % kParticleBytes) != 0u)
        {
            std::ostringstream message;
            message << "Particle cloud binary size must be either 4 + N * 12 bytes for a "
                       "uint32-count-prefixed file or N * 12 bytes for raw float triplets: "
                    << path.string();
            errorMessage = message.str();
            return false;
        }
        particleCount = fileSize / kParticleBytes;
    }

    if (particleCount > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()))
    {
        std::ostringstream message;
        message << "Particle cloud is too large to load on this platform: " << path.string();
        errorMessage = message.str();
        return false;
    }

    particles.reserve(static_cast<std::size_t>(particleCount));
    const unsigned char *data = bytes.data() + dataOffset;
    for (std::uintmax_t i = 0u; i < particleCount; ++i)
    {
        float x                            = 0.0f;
        float y                            = 0.0f;
        float z                            = 0.0f;
        const unsigned char *particleBytes = data + i * kParticleBytes;
        std::memcpy(&x, particleBytes + 0u * sizeof(float), sizeof(float));
        std::memcpy(&y, particleBytes + 1u * sizeof(float), sizeof(float));
        std::memcpy(&z, particleBytes + 2u * sizeof(float), sizeof(float));
        particles.push_back(Diligent::float3{x, y, z});
    }

    return true;
}

std::vector<Diligent::float3> loadParticleCloud(const std::filesystem::path &path)
{
    std::vector<Diligent::float3> particles;
    std::string errorMessage;
    if (!readParticleCloudBin(path, particles, errorMessage))
    {
        CRESSIM_LOG_ERROR(errorMessage);
        particles.clear();
    }
    return particles;
}

} // namespace cressim::neo::physics
